import logging


LABEL_MAPPING = {
            0: 0,       # B-PS
            1: 0,
            2: 1,       # B-LC
            3: 1,
            4: 2,       # B-OG
            5: 2,
            6: 3,       # B-DT
            7: 3,
            8: 4,       # B-IT
            9: 4,
            10: 5,      # B-QT
            11: 5,
            12: 6,      # O
            -100: -100, # LABEL_PAD_TOKEN
        }


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)