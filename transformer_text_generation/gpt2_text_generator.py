import argparse

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


MODEL_NAME = "gpt2"


def load_tokenizer():
    # Loading tokenizer: GPT-2 tokenizer converts text into numerical token IDs.
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    # GPT-2 does not define a padding token, so we use the end-of-text token.
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model():
    # Loading Transformer model: GPT-2 is a GPT-style decoder-only architecture.
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.eval()
    return model


def prepare_prompt(user_prompt):
    return (
        "Write a complete educational paragraph about the following topic.\n"
        f"Topic: {user_prompt.strip()}\n"
        "Paragraph:"
    )


def clean_paragraph(text):
    paragraph = text.split("Paragraph:", 1)[-1].strip()
    paragraph = " ".join(paragraph.split())

    if paragraph and paragraph[-1] not in ".!?":
        last_stop = max(paragraph.rfind("."), paragraph.rfind("!"), paragraph.rfind("?"))
        if last_stop > 50:
            paragraph = paragraph[: last_stop + 1]
        else:
            paragraph = f"{paragraph}."

    return paragraph


def generate_paragraph(
    prompt,
    tokenizer,
    model,
    max_length=180,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
):
    formatted_prompt = prepare_prompt(prompt)

    # Tokenization: convert the user prompt into model-readable PyTorch tensors.
    encoded_input = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    # Text generation: GPT-2 predicts the next tokens using sampling controls.
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=encoded_input["input_ids"],
            attention_mask=encoded_input["attention_mask"],
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decoding output: convert generated token IDs back into readable text.
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return clean_paragraph(generated_text)


def parse_args():
    parser = argparse.ArgumentParser(description="GPT-2 Transformer Text Generator")
    parser.add_argument("--max_length", type=int, default=180, help="Maximum total token length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Controls randomness")
    parser.add_argument("--top_k", type=int, default=50, help="Limits sampling to top K tokens")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling probability")
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = load_tokenizer()
    model = load_model()

    print("GPT-2 Transformer Text Generation")
    print("Type a topic or starting sentence. Type 'exit' to quit.\n")

    while True:
        user_prompt = input("Enter prompt: ").strip()

        if user_prompt.lower() in {"exit", "quit"}:
            print("Program closed.")
            break

        if not user_prompt:
            print("Please enter a valid topic or starting sentence.\n")
            continue

        paragraph = generate_paragraph(
            prompt=user_prompt,
            tokenizer=tokenizer,
            model=model,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        print("\nGenerated Paragraph:")
        print(paragraph)
        print()


if __name__ == "__main__":
    main()
