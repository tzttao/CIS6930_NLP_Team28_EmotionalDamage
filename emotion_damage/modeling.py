"""Model helpers for training, evaluation and inference."""

from __future__ import annotations

import os
import time
from typing import List, Sequence

import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Pad, Tuple
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

from .labels import ID2LABEL, LABELS
from .metrics import MultiLabelReport


def create_model(model_name: str):
    """Initializes the ERNIE encoder and tokenizer."""

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_classes=len(LABELS)
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def create_optimizer(model, learning_rate: float):
    return paddle.optimizer.AdamW(
        learning_rate=learning_rate, parameters=model.parameters(), weight_decay=0.01
    )


def create_loss_fn():
    return paddle.nn.BCEWithLogitsLoss()


def evaluate(model, criterion, metric, data_loader):
    model.eval()
    metric.reset()
    losses = []
    with paddle.no_grad():
        for batch in data_loader:
            logits = model(batch["input_ids"], batch["token_type_ids"])
            loss = criterion(logits, batch["labels"])
            probs = F.sigmoid(logits)
            metric.update(probs, batch["labels"])
            losses.append(float(loss.numpy()))
    auc, f1_score, precision, recall = metric.accumulate()
    model.train()
    metric.reset()
    return float(np.mean(losses)), auc, f1_score, precision, recall


def maybe_save_checkpoint(model, tokenizer, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def train_loop(
    model,
    tokenizer,
    train_loader,
    eval_loader,
    epochs: int,
    learning_rate: float,
    checkpoint_dir: str,
    log_steps: int = 10,
    eval_steps: int = 40,
):
    optimizer = create_optimizer(model, learning_rate)
    criterion = create_loss_fn()
    metric = MultiLabelReport()

    global_step = 0
    best_f1 = 0.0
    tic = time.time()
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(train_loader, start=1):
            logits = model(batch["input_ids"], batch["token_type_ids"])
            loss = criterion(logits, batch["labels"])
            probs = F.sigmoid(logits)
            metric.update(probs, batch["labels"])

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            global_step += 1
            if global_step % log_steps == 0:
                auc, f1_score, _, _ = metric.accumulate()
                speed = log_steps / (time.time() - tic)
                print(
                    f"global step {global_step}, epoch {epoch}, batch {step}, "
                    f"loss {float(loss.numpy()):.5f}, auc {auc:.5f}, f1 {f1_score:.5f}, speed {speed:.2f} step/s"
                )
                tic = time.time()
                metric.reset()

            if eval_loader and global_step % eval_steps == 0:
                eval_loss, _, eval_f1, _, _ = evaluate(model, criterion, metric, eval_loader)
                print(f"[eval] step {global_step}, loss {eval_loss:.5f}, f1 {eval_f1:.5f}")
                if eval_f1 > best_f1:
                    best_f1 = eval_f1
                    maybe_save_checkpoint(model, tokenizer, checkpoint_dir)

    return {"best_f1": best_f1, "global_step": global_step}


def load_checkpoint(model, checkpoint_dir: str):
    state_path = os.path.join(checkpoint_dir, "model_state.pdparams")
    if not os.path.exists(state_path):
        raise FileNotFoundError(
            f"Could not find model_state.pdparams in {checkpoint_dir}. "
            "Run train.py first or point to a directory created by PaddleNLP."
        )
    state_dict = paddle.load(state_path)
    model.set_dict(state_dict)


def predict_texts(
    model,
    tokenizer,
    texts: Sequence[str],
    threshold: float = 0.5,
    batch_size: int = 8,
    max_seq_length: int = 64,
):
    model.eval()
    batches = []
    current_batch = []
    for text in texts:
        encoded = tokenizer(text=text, max_seq_len=max_seq_length)
        current_batch.append((encoded["input_ids"], encoded["token_type_ids"]))
        if len(current_batch) == batch_size:
            batches.append(current_batch)
            current_batch = []
    if current_batch:
        batches.append(current_batch)

    pad_token_id = getattr(tokenizer, "pad_token_id", 0)
    pad_token_type_id = getattr(tokenizer, "pad_token_type_id", 0)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=pad_token_id),
        Pad(axis=0, pad_val=pad_token_type_id),
    ): fn(samples)

    predictions: List[List[str]] = []
    with paddle.no_grad():
        for batch in batches:
            input_ids, token_type_ids = batchify_fn(batch)
            logits = model(paddle.to_tensor(input_ids), paddle.to_tensor(token_type_ids))
            probs = F.sigmoid(logits).numpy()
            for example_probs in probs:
                labels = [
                    ID2LABEL[idx]
                    for idx, score in enumerate(example_probs)
                    if score >= threshold
                ]
                predictions.append(labels)
    return predictions
