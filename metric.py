import utils
import logging
import numpy as np

import evaluate
from seqeval.metrics import classification_report

metric = evaluate.load("seqeval")
logger = logging.getLogger(__name__)


def compute_metrics(p, return_entity_level_metrics=False):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    label_list = utils.get_labels()
    id_to_label = {v: k for k, v in label_list.items()}

    # Remove ignored index (special tokens)
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    logger.info(f"\n {classification_report(true_labels, true_predictions, suffix=False)}")

    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
