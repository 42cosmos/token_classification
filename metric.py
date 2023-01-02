import utils
import logging
import numpy as np
from collections import defaultdict
from typing import List, Dict, Sequence

import evaluate
from seqeval.metrics import classification_report, performance_measure

metric = evaluate.load("seqeval")
logger = logging.getLogger(__name__)


def decompress_labels(x: list):
    label_list = [
        "B-DT", "B-LC", "B-OG", "B-PS", "B-QT", "B-TI", "B-DUR",
        "O",  # 7
        "I-DT", "I-LC", "I-OG", "I-PS", "I-QT", "I-TI", "I-DUR",
    ]

    result = []
    before_num = 999
    for idx, num in enumerate(x):
        if num != -100:
            result_num = num
            if num != -100 and num != 7:
                if before_num == num:
                    bumper = 8 # num_labels
                    result_num += bumper
            result.append(label_list[result_num])
            before_num = num
    return result


def compute_metrics(p, compress=False, inference=False, save_result=True):
    predictions, labels = p

    if not inference:
        predictions = np.argmax(predictions, axis=2)

    label_list = utils.get_labels(compress)
    id_to_label = {v: k for k, v in label_list.items()}

    if compress:
        true_predictions = [
            decompress_labels(y)
            for (x, y) in zip(predictions, labels)
        ]

        true_labels = [
            decompress_labels(y)
            for (x, y) in zip(predictions, labels)
        ]

    else:
        true_predictions = [
            [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        true_labels = [
            [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    tagging_inform = extract_tp_actual_correct(y_pred=true_predictions, y_true=true_labels)

    for k, v in results.items():
        try:
            v.update(tagging_inform[k])
        except:
            pass

    logger.info(f"\n {classification_report(true_labels, true_predictions, suffix=False)}")
    for k, v in results.items():
        logger.info(f"\n{k}: {v}")

    if save_result:
        with open("result.txt", "w") as f:
            f.write(f"\n {classification_report(true_labels, true_predictions, suffix=False)}")
            for k, v in results.items():
                f.write(f"\n{k}: {v}")

    return results if inference else {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def start_of_chunk(prev_tag, tag, prev_type, type_):
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'E' and tag == 'E':
        chunk_start = True
    if prev_tag == 'E' and tag == 'I':
        chunk_start = True
    if prev_tag == 'S' and tag == 'E':
        chunk_start = True
    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'E':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def end_of_chunk(prev_tag, tag, prev_type, type_):
    chunk_end = False

    if prev_tag == 'E':
        chunk_end = True
    if prev_tag == 'S':
        chunk_end = True

    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def get_entities(seq, suffix=False):
    def _validate_chunk(chunk, suffix):
        if chunk in ['O', 'B', 'I', 'E', 'S']:
            return

        if suffix:
            if not chunk.endswith(('-B', '-I', '-E', '-S')):
                warnings.warn('{} seems not to be NE tag.'.format(chunk))

        else:
            if not chunk.startswith(('B-', 'I-', 'E-', 'S-')):
                warnings.warn('{} seems not to be NE tag.'.format(chunk))

    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        _validate_chunk(chunk, suffix)

        if suffix:
            tag = chunk[-1]
            type_ = chunk[:-1].rsplit('-', maxsplit=1)[0] or '_'
        else:
            tag = chunk[0]
            type_ = chunk[1:].split('-', maxsplit=1)[-1] or '_'

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def extract_tp_actual_correct(y_true, y_pred, suffix=False, *args):
    entities_true = defaultdict(set)
    entities_pred = defaultdict(set)
    for type_name, start, end in get_entities(y_true, suffix):
        entities_true[type_name].add((start, end))
    for type_name, start, end in get_entities(y_pred, suffix):
        entities_pred[type_name].add((start, end))

    target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))

    target_dict = {}

    tp_sum = np.array([], dtype=np.int32)
    pred_sum = np.array([], dtype=np.int32)
    true_sum = np.array([], dtype=np.int32)
    for type_name in target_names:
        entities_true_type = entities_true.get(type_name, set())
        entities_pred_type = entities_pred.get(type_name, set())
        tp_sum = len(entities_true_type & entities_pred_type)
        pred_sum = len(entities_pred_type)
        true_sum = len(entities_true_type)

        true_posi = tp_sum
        false_posi = pred_sum - tp_sum
        false_nega = true_sum - tp_sum

        target_dict[type_name] = {"TP": true_posi, "FP": false_posi, "FN": false_nega}
        # {"tp_sum": tp_sum, "pred_sum": pred_sum, "true_sum": true_sum}
        #

    return target_dict
