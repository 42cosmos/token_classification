import os
import logging

LABEL_MAPPING = {
    0: 0,  # B-DT
    1: 0,
    2: 1,  # B-LC
    3: 1,
    4: 2,  # B-OG
    5: 2,
    6: 3,  # B-PS
    7: 3,
    8: 4,  # B-QT
    9: 4,
    10: 5,  # B-TI
    11: 5,
    12: 6,  # O
    -100: -100,  # LABEL_PAD_TOKEN
}


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def get_test_texts(args):
    texts = []
    with open(os.path.join(args.data_dir, args.test_file), 'r', encoding='utf-8') as f:
        for line in f:
            text, _ = line.split('\t')
            text = text.split()
            texts.append(text)

    return texts


def get_labels():
    label_raw = """B-DT(0)
    I-DT(1)
    B-LC(2)
    I-LC(3)
    B-OG(4)
    I-OG(5)
    B-PS(6)
    I-PS(7)
    B-QT(8)
    I-QT(9)
    B-TI(10)
    I-TI(11)
    O(12)
    B-DUR(13)
    I-DUR(14)""".split("\n")
    result = {label.split("(")[0].strip(): int(label.split("(")[1].replace(")", "")) for label in label_raw}
    return result
