#!/usr/bin/env python3

"""Runs inference against the fine-tuned ERNIE checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import paddle

from emotion_damage.data import clean_text
from emotion_damage.modeling import create_model, load_checkpoint, predict_texts


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="ernie_ckpt", help="Directory with saved Paddle weights.")
    parser.add_argument("--model-name", default="ernie-3.0-medium-zh", help="Base ERNIE checkpoint.")
    parser.add_argument("--max-length", type=int, default=64, help="Maximum length for tokenized inputs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size used during inference.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification probability threshold.")
    parser.add_argument(
        "--texts",
        nargs="*",
        help="Optional list of raw texts. When omitted the script expects ``--input-file``.",
    )
    parser.add_argument(
        "--input-file",
        help="Plain text file with one example per line. Ignored when ``--texts`` is specified.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to store the JSON predictions instead of only printing them.",
    )
    return parser.parse_args()


def read_texts(args) -> list[str]:
    if args.texts:
        return [clean_text(text) for text in args.texts]
    if args.input_file:
        with open(args.input_file, encoding="utf-8") as handle:
            return [clean_text(line.strip()) for line in handle if line.strip()]
    return [
        "You do a great job!",
        "I am so disappointed by everything you have done.",
        "I am so nervous about the exam tomorrow.",
    ]


def main():
    args = parse_args()
    device = "gpu:0" if paddle.is_compiled_with_cuda() else "cpu"
    paddle.set_device(device)

    texts = read_texts(args)
    if not texts:
        raise SystemExit("No input texts were provided.")

    model, tokenizer = create_model(args.model_name)
    load_checkpoint(model, args.checkpoint_dir)
    predictions = predict_texts(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        threshold=args.threshold,
        batch_size=args.batch_size,
        max_seq_length=args.max_length,
    )

    payload = [
        {"text": text, "labels": labels if labels else ["neutral"], "scores": ">= {:.2f}".format(args.threshold)}
        for text, labels in zip(texts, predictions)
    ]
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Predictions saved to {args.output}")
    else:
        for item in payload:
            print(f"{item['text']} -> {', '.join(item['labels']) or 'neutral'}")


if __name__ == "__main__":
    main()
