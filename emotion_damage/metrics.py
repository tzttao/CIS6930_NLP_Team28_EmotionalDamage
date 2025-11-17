"""Evaluation metrics for multi-label classification."""

from __future__ import annotations

import numpy as np
from paddle.metric import Metric
from sklearn.metrics import f1_score as sk_f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score


class MultiLabelReport(Metric):
    """Computes AUC and F1 with an adaptive threshold search."""

    def __init__(self, name: str = "MultiLabelReport", average: str = "micro"):
        super().__init__()
        self.average = average
        self._name = name
        self.reset()

    def _f1_score(self, y_prob: np.ndarray):
        best_score = 0.0
        best_precision = 0.0
        best_recall = 0.0
        for threshold in [i * 0.01 for i in range(100)]:
            preds = (y_prob > threshold).astype("int32")
            score = sk_f1_score(
                y_true=self.y_true, y_pred=preds, average=self.average, zero_division=0
            )
            if score > best_score:
                best_score = float(score)
                best_precision = float(
                    precision_score(
                        y_true=self.y_true, y_pred=preds, average=self.average, zero_division=0
                    )
                )
                best_recall = float(
                    recall_score(
                        y_true=self.y_true, y_pred=preds, average=self.average, zero_division=0
                    )
                )
        return best_score, best_precision, best_recall

    def reset(self):
        self.y_prob = None
        self.y_true = None

    def update(self, probs, labels):
        probs_np = probs.numpy()
        labels_np = labels.numpy()
        if self.y_prob is not None:
            self.y_prob = np.append(self.y_prob, probs_np, axis=0)
            self.y_true = np.append(self.y_true, labels_np, axis=0)
        else:
            self.y_prob = probs_np
            self.y_true = labels_np

    def accumulate(self):
        auc = roc_auc_score(y_score=self.y_prob, y_true=self.y_true, average=self.average)
        f1_score, precision, recall = self._f1_score(self.y_prob)
        return auc, f1_score, precision, recall

    def name(self):
        return self._name
