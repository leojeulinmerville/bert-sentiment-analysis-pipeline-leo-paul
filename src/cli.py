import argparse

from src.inference import load_model, predict_sentiment


def parse_args():
    parser = argparse.ArgumentParser(description="Predict sentiment for a text using BERT.")
    parser.add_argument(
        "--text",
        type=str,
        default="I love this app!",
        help="Text to classify.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./model_out",
        help="Path to fine-tuned model directory (mount as volume if outside the image).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="Hugging Face model id used for tokenizer and fallback weights.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer, model = load_model(model_path=args.model_path, name=args.model_name)
    label = predict_sentiment(args.text, tokenizer, model)
    print(f"Text: {args.text}")
    print(f"Predicted label: {label}")


if __name__ == "__main__":
    main()
