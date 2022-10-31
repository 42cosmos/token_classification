import numpy as np
from datasets import load_metric
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import wandb

import utils


# def compute_metrics(eval_preds):
#     label_names = utils.get_labels()
#     metric = load_metric("seqeval")
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)
#
#     # 무시된 인덱스(특수 토큰들)를 제거하고 레이블로 변환
#     true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
#     true_predictions = [
#         [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]
#     all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
#     return {
#         "precision": all_metrics["overall_precision"],
#         "recall": all_metrics["overall_recall"],
#         "f1": all_metrics["overall_f1"],
#         "accuracy": all_metrics["overall_accuracy"],
#     }
#


def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    return f1_pre_rec(labels, preds)


def f1_pre_rec(labels, preds):
    results = {
        "precision": precision_score(labels, preds, suffix=True),
        "recall": recall_score(labels, preds, suffix=True),
        "f1": f1_score(labels, preds, suffix=True)
    }
    # wandb.log(results)
    return results


def show_report(labels, preds):
    return classification_report(labels, preds, suffix=True)
