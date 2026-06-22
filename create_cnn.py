import argparse
import json
import os

import numpy as np

BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_gender_model.keras")
META_PATH  = os.path.join(MODEL_DIR, "cnn_meta.json")

#  Actual dataset path 
DATASET_PATH = os.path.join(BASE_DIR, "cnn_datasets", "testdata", "testdata")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--img-size",   type=int,   default=128)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--augment",    action="store_true")
    p.add_argument("--fine-tune",  action="store_true")
    return p.parse_args()



def build_model(img_size, lr):
    import tensorflow as tf

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
 
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model, backbone


def unfreeze_top(model, backbone, lr):
    import tensorflow as tf

    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr / 10),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    print("  Fine-tune: last 30 backbone layers unlocked.")




def build_dataset(img_size, batch_size, augment):
   
    import tensorflow as tf

    print(f"\n  Dataset path : {DATASET_PATH}")

    # Check folders exist
    men_dir   = os.path.join(DATASET_PATH, "men")
    women_dir = os.path.join(DATASET_PATH, "women")
    if not os.path.isdir(men_dir) or not os.path.isdir(women_dir):
        raise FileNotFoundError(
            f"Expected  men/  and  women/  inside:\n  {DATASET_PATH}\n"
            "Please check your folder structure."
        )

    common = dict(
        label_mode="binary",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        seed=42,
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        subset="training",
        validation_split=0.15,
        **common,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        subset="validation",
        validation_split=0.15,
        **common,
    )

    # TF alphabetical order: ['men', 'women'] → men=0, women=1
    class_names = train_ds.class_names
    print(f"  Class names  : {class_names}  (TF order: index 0 = {class_names[0]}, 1 = {class_names[1]})")

    n_train = sum(1 for _ in train_ds) * batch_size
    n_val   = sum(1 for _ in val_ds)   * batch_size
    print(f"  Train batches: ~{n_train} images")
    print(f"  Val   batches: ~{n_val}  images\n")

    # Normalise to [0,1]
    rescale = lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)
    train_ds = train_ds.map(rescale, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds   = val_ds.map(rescale, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.15),
        ])
        train_ds = train_ds.map(
            lambda x, y: (aug(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(500).prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names



def make_callbacks():
    import tensorflow as tf

    return [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.4,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]



def main():
    args = parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 55)
    print("  Gender CNN  —  MobileNetV2 Transfer Learning")
    print("=" * 55)
    print(f"  Image size   : {args.img_size}×{args.img_size}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Augmentation : {args.augment}")
    print(f"  Fine-tune    : {args.fine_tune}")

    train_ds, val_ds, class_names = build_dataset(
        args.img_size, args.batch_size, args.augment
    )

    model, backbone = build_model(args.img_size, args.lr)
    print(f"\n  Trainable params : {model.count_params():,}\n")

    # ── Phase 1: train head only 
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=make_callbacks(),
        verbose=1,
    ).history

    # ── Phase 2: optional fine-tune 
    if args.fine_tune:
        print("\n  ── Fine-tuning phase ──")
        unfreeze_top(model, backbone, args.lr)
        ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=max(5, args.epochs // 2),
            callbacks=make_callbacks(),
            verbose=1,
        ).history
        for k, v in ft.items():
            history.setdefault(k, []).extend(v)

    #  Saving model & metadata 
    model.save(MODEL_PATH)  # .keras format — no TrueDivide issue

    meta = {
        "img_size":     args.img_size,
        "class_names":  class_names,          # e.g. ['men', 'women']
        "demo_mode":    False,
        "val_accuracy": float(max(history.get("val_accuracy", [0]))),
        "val_auc":      float(max(history.get("val_auc",      [0]))),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Model saved  → {MODEL_PATH}")
    print(f"  Meta  saved  → {META_PATH}")
    print(f"\n  Best val accuracy : {meta['val_accuracy']*100:.1f}%")
    print(f"  Best val AUC      : {meta['val_auc']:.4f}")
    print("\n  Done. Start Flask:  python app.py\n")


if __name__ == "__main__":
    main()