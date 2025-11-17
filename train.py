#!/usr/bin/env python3

"""Command line training utility for the Emotion Damage classifier."""

from __future__ import annotations

import argparse
import paddle

from emotion_damage.data import build_data_loader
from emotion_damage.metrics import MultiLabelReport
from emotion_damage.modeling import create_model, evaluate, load_checkpoint, train_loop


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", default="train.csv", help="Path to the training TSV file.")
    parser.add_argument("--eval-data", default="test.csv", help="Path to the evaluation TSV file.")
    parser.add_argument("--model-name", default="ernie-3.0-medium-zh", help="Pretrained ERNIE checkpoint.")
    parser.add_argument("--epochs", type=int, default=3, help="Total number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=5e-3, help="AdamW learning rate.")
    parser.add_argument("--checkpoint-dir", default="ernie_ckpt", help="Directory used to store model weights.")
    parser.add_argument("--max-length", type=int, default=64, help="Maximum sequence length.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size used for training.")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Batch size used for evaluation.")
    parser.add_argument("--log-steps", type=int, default=10, help="How often to log training metrics.")
    parser.add_argument("--eval-steps", type=int, default=40, help="How often to run evaluation.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from ``--checkpoint-dir`` if weights already exist.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional Paddle device identifier, e.g. ``gpu:0``. Defaults to GPU if available.",
    )
    return parser.parse_args()


def set_device(device_flag: str | None):
    if device_flag:
        paddle.set_device(device_flag)
        return device_flag
    if paddle.is_compiled_with_cuda():
        device_flag = "gpu:0"
    else:
        device_flag = "cpu"
    paddle.set_device(device_flag)
    return device_flag


def main():
    args = parse_args()
    device = set_device(args.device)
    print(f"Using device: {device}")

    model, tokenizer = create_model(args.model_name)
    if args.resume:
        try:
            load_checkpoint(model, args.checkpoint_dir)
            print(f"Loaded weights from {args.checkpoint_dir}")
        except FileNotFoundError:
            print("No checkpoint found, starting from scratch.")

    _, train_loader = build_data_loader(
        filepath=args.train_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_length=args.max_length,
        shuffle=True,
    )
    eval_loader = None
    if args.eval_data:
        _, eval_loader = build_data_loader(
            filepath=args.eval_data,
            tokenizer=tokenizer,
            batch_size=args.eval_batch_size,
            max_seq_length=args.max_length,
            shuffle=False,
        )

    stats = train_loop(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        log_steps=args.log_steps,
        eval_steps=args.eval_steps,
    )
    print(f"Training complete. Best F1: {stats['best_f1']:.5f}")

    if eval_loader is not None:
        criterion = paddle.nn.BCEWithLogitsLoss()
        metric = MultiLabelReport()
        eval_loss, auc, f1_score, precision, recall = evaluate(model, criterion, metric, eval_loader)
        print(
            "Final evaluation -- "
            f"loss {eval_loss:.5f}, auc {auc:.5f}, f1 {f1_score:.5f}, "
            f"precision {precision:.5f}, recall {recall:.5f}"
        )


if __name__ == "__main__":
    main()
