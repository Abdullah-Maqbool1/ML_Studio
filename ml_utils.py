import json
import os
from functools import lru_cache
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler


BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_gender_model.keras")
META_PATH  = os.path.join(MODEL_DIR, "cnn_meta.json")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")

QA_CONTEXT = """
Applied Machine Learning is a practical course where students learn how to
prepare datasets, train models, evaluate predictions, and deploy intelligent
systems. This Flask application demonstrates clustering, association rule
mining, computer vision, speech recognition, sentiment analysis, question
answering, text generation, translation, and named entity recognition.
"""


# ──────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────

def _html_table(df, max_rows=25):
    return df.head(max_rows).to_html(
        classes="min-w-full divide-y divide-slate-200 text-sm",
        index=False,
        border=0,
        escape=False,
    )


def _create_cluster_plot(numeric_df, labels, algorithm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filename    = f"{algorithm}_clusters_{uuid4().hex}.png"
    output_path = os.path.join(UPLOAD_DIR, filename)

    plot_df = numeric_df.reset_index(drop=True)
    x_values = plot_df.iloc[:, 0]
    x_label  = plot_df.columns[0]

    if plot_df.shape[1] >= 2:
        y_values = plot_df.iloc[:, 1]
        y_label  = plot_df.columns[1]
    else:
        y_values = plot_df.index
        y_label  = "Row Index"

    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(
        x_values, y_values,
        c=labels, cmap="viridis", s=70, alpha=0.85,
        edgecolors="white", linewidths=0.7,
    )
    plt.title(f"{algorithm.upper()} Cluster Visualization")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, linestyle="--", alpha=0.3)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return f"uploads/{filename}"


# ──────────────────────────────────────────────────────────────
# Clustering
# ──────────────────────────────────────────────────────────────

def run_clustering(file_path, algorithm):
    df      = pd.read_csv(file_path)
    numeric = df.select_dtypes(include=[np.number]).dropna()

    if numeric.empty:
        raise ValueError("The uploaded CSV must contain at least one numeric column.")

    scaled = StandardScaler().fit_transform(numeric)

    if algorithm == "dbscan":
        labels = DBSCAN(eps=0.8, min_samples=3).fit_predict(scaled)
    elif algorithm == "kmeans":
        n_clusters = min(3, len(numeric))
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(scaled)
    else:
        raise ValueError("Unsupported clustering algorithm.")

    output = df.loc[numeric.index].copy()
    output["Cluster"] = labels
    counts = (
        output["Cluster"]
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="Rows")
    )
    plot_filename = _create_cluster_plot(numeric, labels, algorithm)

    return {
        "type":    "table_group",
        "summary": f"Processed {len(output)} usable rows with {numeric.shape[1]} numeric feature(s).",
        "plots":   [{"title": "Cluster Scatter Plot", "src": plot_filename}],
        "tables":  [
            {"title": "Cluster Counts",   "html": _html_table(counts)},
            {"title": "Clustered Sample", "html": _html_table(output)},
        ],
    }


# ──────────────────────────────────────────────────────────────
# CNN — Gender Classification  (MobileNetV2 real-world model)
# ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_cnn_model():
    import tensorflow as tf

    # Support both .keras (new) and .h5 (legacy) formats
    keras_path = os.path.join(MODEL_DIR, "cnn_gender_model.keras")
    h5_path    = os.path.join(MODEL_DIR, "cnn_gender_model.h5")

    if os.path.exists(keras_path) and os.path.getsize(keras_path) > 0:
        path = keras_path
    elif os.path.exists(h5_path) and os.path.getsize(h5_path) > 0:
        path = h5_path
    else:
        raise FileNotFoundError(
            "CNN model not found. Run  py -3.10 create_cnn.py  first."
        )

    return tf.keras.models.load_model(path, compile=False)


@lru_cache(maxsize=1)
def _load_cnn_meta() -> dict:
    """
    Returns metadata written by create_cnn.py:
        img_size     : int   — model input size (default 128)
        class_names  : list  — TF alphabetical order e.g. ['men', 'women']
        val_accuracy : float
        demo_mode    : bool
    """
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            return json.load(f)
    # Safe defaults matching create_cnn.py alphabetical output
    return {
        "img_size":    128,
        "class_names": ["men", "women"],
        "demo_mode":   True,
        "val_accuracy": 0.0,
    }


def _preprocess_image(image_path: str, img_size: int) -> np.ndarray:
    """Resize + normalise to [0,1], return (1, H, W, 3)."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def _image_quality_warning(image_path: str) -> str | None:
    """Return a warning string if image looks unusable, else None."""
    from PIL import Image, ImageStat
    try:
        img  = Image.open(image_path).convert("RGB")
        w, h = img.size
        if w < 48 or h < 48:
            return "Image is very small — classification may be unreliable."
        stat     = ImageStat.Stat(img)
        mean_lum = sum(stat.mean)   / 3.0
        std_lum  = sum(stat.stddev) / 3.0
        if mean_lum < 15:
            return "Image appears very dark — results may be unreliable."
        if std_lum < 5:
            return "Image has very low contrast — results may be unreliable."
    except Exception:
        pass
    return None


def classify_image(image_path: str) -> dict:
    """
    Classify a face image as Male or Female.

    TF image_dataset_from_directory sorts folders alphabetically:
        ['men', 'women']  →  men=0, women=1

    The sigmoid output is therefore:
        score → 0  =  men  (Male)
        score → 1  =  women (Female)

    So we flip the usual convention:
        score < 0.5  →  Male   (class 0 = men)
        score ≥ 0.5  →  Female (class 1 = women)
    """
    meta      = _load_cnn_meta()
    img_size  = meta.get("img_size", 128)
    demo_mode = meta.get("demo_mode", True)
    val_acc   = meta.get("val_accuracy", 0.0)
    # class_names[0] is alphabetically first: 'men'
    # class_names[1] is alphabetically second: 'women'
    class_names = meta.get("class_names", ["men", "women"])

    model  = _load_cnn_model()
    tensor = _preprocess_image(image_path, img_size)
    score  = float(model.predict(tensor, verbose=0)[0][0])

    # Alphabetical: men=0 (score near 0), women=1 (score near 1)
    if score < 0.5:
        label      = "Male"
        confidence = 1.0 - score          # distance from 0
    else:
        label      = "Female"
        confidence = score                 # distance from 1

    warning = _image_quality_warning(image_path)


    conf_pct = round(confidence * 100, 2)

    result = {
        "type":           "prediction",
        "label":          label,
        "confidence":     f"{conf_pct:.2f}%",
        "confidence_pct": conf_pct,
        "raw_score":      round(score, 5)
    }
    if warning:
        result["warning"] = warning

    return result


# ──────────────────────────────────────────────────────────────
# Sentiment (voice)
# ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _sentiment_pipeline():
    from transformers import pipeline
    return pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    )


# ──────────────────────────────────────────────────────────────
# QA (voice)
# ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _qa_model():
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    model_name = "distilbert-base-cased-distilled-squad"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model


# ──────────────────────────────────────────────────────────────
# Text generation
# ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _text_generation_pipeline():
    from transformers import pipeline
    return pipeline("text-generation", model="gpt2")


@lru_cache(maxsize=1)
def _instruction_generation_model():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    model_name = "google/flan-t5-small"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


# ──────────────────────────────────────────────────────────────
# Translation
# ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _translation_model():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    model_name = "Helsinki-NLP/opus-mt-en-ur"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


# ──────────────────────────────────────────────────────────────
# NER
# ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _ner_pipeline():
    from transformers import pipeline
    return pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",
    )


# ──────────────────────────────────────────────────────────────
# Audio helpers
# ──────────────────────────────────────────────────────────────

def audio_to_text(audio_path):
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError as exc:
        raise ValueError("Speech could not be understood. Try a clearer audio file.") from exc
    except sr.RequestError as exc:
        raise ValueError(f"Speech recognition service error: {exc}") from exc


def text_to_audio(text, output_filename=None):
    from gtts import gTTS
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filename    = output_filename or f"answer_{uuid4().hex}.mp3"
    output_path = os.path.join(UPLOAD_DIR, filename)
    gTTS(text=text, lang="en").save(output_path)
    return output_path


# ──────────────────────────────────────────────────────────────
# Public service functions
# ──────────────────────────────────────────────────────────────

def sentiment_from_voice(audio_path):
    text      = audio_to_text(audio_path)
    sentiment = _sentiment_pipeline()(text)[0]
    return {
        "type":          "sentiment",
        "transcription": text,
        "label":         sentiment["label"],
        "score":         f"{sentiment['score'] * 100:.2f}%",
    }


def answer_question_from_voice(audio_path):
    question    = audio_to_text(audio_path)
    answer      = answer_question(question)
    answer_text = answer["answer"]
    audio_file  = text_to_audio(answer_text)
    return {
        "type":       "qa_audio",
        "question":   question,
        "answer":     answer_text,
        "score":      f"{answer['score'] * 100:.2f}%",
        "audio_file": audio_file,
    }


def answer_question(question):
    import torch
    tokenizer, model = _qa_model()
    inputs = tokenizer(
        question, QA_CONTEXT,
        return_tensors="pt", truncation=True, max_length=384,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    start_index = int(torch.argmax(outputs.start_logits))
    end_index   = int(torch.argmax(outputs.end_logits))
    if end_index < start_index:
        end_index = start_index

    answer_ids  = inputs["input_ids"][0][start_index: end_index + 1]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    start_score = torch.softmax(outputs.start_logits, dim=-1)[0][start_index]
    end_score   = torch.softmax(outputs.end_logits,   dim=-1)[0][end_index]
    score       = float((start_score * end_score).item())

    if not answer_text:
        answer_text = "This Flask application demonstrates multiple applied machine learning services."

    return {"answer": answer_text, "score": score}


def generate_text(prompt):
    clean_prompt = prompt.strip()
    topic        = clean_prompt
    lowered      = clean_prompt.lower()
    for phrase in (
        "write paragraph on",
        "write a paragraph on",
        "write paragraph about",
        "write a paragraph about",
        "write on",
        "explain",
    ):
        if lowered.startswith(phrase):
            topic = clean_prompt[len(phrase):].strip(" :.-")
            break

    topic = topic.strip() or "machine learning"

    def clean_generated_text(text):
        text = " ".join(text.strip().split())
        if text and text[-1] not in ".!?":
            last_stop = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
            if last_stop > 40:
                text = text[:last_stop + 1]
        return text

    def is_good_paragraph(text):
        if not text:
            return False
        words = text.split()
        if len(words) < 35:
            return False
        if text[-1] not in ".!?":
            return False
        if text.startswith(('"', "'", "Introduction to", "Part ")):
            return False
        if ":" in text[:45]:
            return False
        lowered_text = text.lower()
        weak_patterns = (
            "write and write", "learn and write", "computers, computers",
            "computer program, a software program", "telephonics", "telegrams",
        )
        if any(p in lowered_text for p in weak_patterns):
            return False
        cleaned_words = [
            w.strip(".,:;!?()[]'\"").lower()
            for w in words if len(w.strip(".,:;!?()[]'\"")) > 3
        ]
        if cleaned_words and len(set(cleaned_words)) / len(cleaned_words) < 0.58:
            return False
        topic_words = [w.strip(".,:;!?()[]").lower() for w in topic.split() if len(w) > 3]
        if topic_words and not any(w in lowered_text for w in topic_words[:4]):
            return False
        topic_lower = topic.lower()
        if "machine learning" in topic_lower or "learning" in topic_lower:
            return any(w in lowered_text for w in ("data", "model", "prediction", "algorithm", "classification"))
        if "network" in topic_lower:
            return any(w in lowered_text for w in ("device", "protocol", "routing", "data", "internet", "server"))
        return True

    def fallback_paragraph():
        topic_lower = topic.lower()
        if "machine learning" in topic_lower or "learning" in topic_lower:
            return (
                "Machine learning helps students understand how computers can learn from examples and make predictions from data. "
                "It builds practical skills in data preparation, model training, evaluation, and problem solving. "
                "Students can use machine learning in projects such as image classification, sentiment analysis, recommendation systems, "
                "and prediction tasks. Learning this subject also improves logical thinking because students must compare results, reduce errors, "
                "and choose suitable algorithms for real-world problems."
            )
        if "network" in topic_lower:
            subject = topic.title()
            return (
                f"{subject} is an important area of computer science that explains how devices exchange data and share resources. "
                "It includes concepts such as protocols, IP addresses, routing, bandwidth, security, and client-server communication. "
                "Students study this topic to understand how the internet, email, cloud services, and web applications work behind the scenes. "
                "A strong understanding of networks helps learners design reliable systems and troubleshoot communication problems effectively."
            )
        subject = topic[0].upper() + topic[1:]
        return (
            f"{subject} is an important topic because it helps students connect theoretical knowledge with practical applications. "
            "By studying it carefully, learners can understand its key concepts, real-world uses, advantages, and limitations. "
            "This topic also encourages analytical thinking, problem solving, and clear communication of technical ideas. "
            "With consistent practice, students can apply this knowledge in assignments, projects, and professional work."
        )

    try:
        tokenizer, model = _instruction_generation_model()
        task = (
            "Write one polished academic paragraph of 80 to 110 words. "
            "Do not write a title. Do not use bullet points. "
            f"Topic: {topic}"
        )
        inputs     = tokenizer(task, return_tensors="pt", truncation=True, max_length=256)
        output_ids = model.generate(
            **inputs, max_new_tokens=130, min_new_tokens=55,
            num_beams=4, no_repeat_ngram_size=3, early_stopping=True,
        )
        paragraph = clean_generated_text(tokenizer.decode(output_ids[0], skip_special_tokens=True))
        if is_good_paragraph(paragraph):
            return {"type": "text", "text": paragraph}
    except Exception:
        pass

    instruction_prompt = (
        "Write one clear educational paragraph about the following topic. "
        "Use simple academic English and complete the final sentence.\n\n"
        f"Topic: {topic}\n\nParagraph:"
    )
    try:
        generated = _text_generation_pipeline()(
            instruction_prompt,
            max_new_tokens=90, num_return_sequences=1,
            do_sample=True, temperature=0.55, top_p=0.85,
            repetition_penalty=1.25, pad_token_id=50256,
        )[0]["generated_text"]
        paragraph = generated.split("Paragraph:", 1)[-1].strip()
        paragraph = clean_generated_text(paragraph)
        if not is_good_paragraph(paragraph):
            paragraph = fallback_paragraph()
    except Exception:
        paragraph = fallback_paragraph()

    return {"type": "text", "text": paragraph}


def translate_en_to_ur(text):
    tokenizer, model = _translation_model()
    inputs     = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    output_ids = model.generate(**inputs, max_length=256)
    translated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"type": "text", "text": translated}


def extract_ner(text):
    entities = _ner_pipeline()(text)
    rows = [
        {
            "Entity": e.get("word", ""),
            "Type":   e.get("entity_group", e.get("entity", "")),
            "Score":  f"{float(e.get('score', 0.0)) * 100:.2f}%",
        }
        for e in entities
    ]
    if not rows:
        return {"type": "text", "text": "No named entities were detected."}
    return {
        "type":    "table_group",
        "summary": f"Detected {len(rows)} named entity group(s).",
        "tables":  [{"title": "Entities", "html": _html_table(pd.DataFrame(rows), max_rows=50)}],
    }


# ──────────────────────────────────────────────────────────────
# Apriori
# ──────────────────────────────────────────────────────────────

def _prepare_apriori_dataframe(df):
    from mlxtend.preprocessing import TransactionEncoder

    bool_like = set()
    for col in df.columns:
        unique_values = set(df[col].dropna().astype(str).str.lower().unique())
        bool_like.update(unique_values)

    if bool_like and bool_like.issubset({"0", "1", "true", "false", "yes", "no"}):
        one_hot = df.replace({
            "true": True, "false": False,
            "yes":  True, "no":    False,
            "1":    True, "0":     False,
        })
        return one_hot.astype(bool)

    transactions = []
    for _, row in df.iterrows():
        items = []
        for value in row.dropna().astype(str):
            items.extend([item.strip() for item in value.split(",") if item.strip()])
        if items:
            transactions.append(items)

    if not transactions:
        raise ValueError("CSV must contain transactions or one-hot encoded item columns.")

    encoder = TransactionEncoder()
    encoded = encoder.fit(transactions).transform(transactions)
    return pd.DataFrame(encoded, columns=encoder.columns_)


def run_apriori(file_path):
    from mlxtend.frequent_patterns import apriori, association_rules

    df       = pd.read_csv(file_path)
    one_hot  = _prepare_apriori_dataframe(df)
    frequent = apriori(one_hot, min_support=0.2, use_colnames=True)

    if frequent.empty:
        raise ValueError("No frequent itemsets found. Try a larger transaction dataset.")

    rules = association_rules(frequent, metric="lift", min_threshold=1.0)
    if rules.empty:
        raise ValueError("Frequent itemsets found but no rules met the lift threshold.")

    display = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    display["antecedents"] = display["antecedents"].apply(lambda v: ", ".join(sorted(v)))
    display["consequents"] = display["consequents"].apply(lambda v: ", ".join(sorted(v)))

    return {
        "type":    "table_group",
        "summary": f"Generated {len(rules)} association rule(s).",
        "tables":  [{"title": "Top Association Rules", "html": _html_table(display.sort_values("lift", ascending=False))}],
    }