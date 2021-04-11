import argparse
import sys
import random
from valid.model import Classifier
import re
import pickle
import gzip
import logging
from sklearn.metrics import f1_score


def train_model(train, dev, output, rest):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngram_length", dest="ngram_length", type=int, default=3)
    args, rest = parser.parse_known_args(rest)
    label_counts = {}
    train_instances = train + dev
    logging.info("Computing alphabet size...")
    train_alphabet = set()
    for i in train_instances:
        for v in i["sequence"]:
            train_alphabet.add(v)
    logging.info("Training models...")
    logging.info("Training %d-gram model on train set...", args.ngram_length)
    model = Classifier(order=args.ngram_length, alphabet_size=len(train_alphabet))
    for i in train_instances:
        model.train(i["label"], i["sequence"])
    return pickle.dumps(model)


def apply_model(model, test, args):
    model = pickle.loads(model)
    for i in range(len(test)):
        test[i]["scores"] = model.probabilities(test[i]["sequence"])
    return test
