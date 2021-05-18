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
        for tok in i["tokens"]:
            for v in tok["form"]:
                train_alphabet.add(v)
    logging.info("Training models...")
    logging.info("Training %d-gram model on train set...", args.ngram_length)
    word_model = Classifier(order=args.ngram_length, alphabet_size=256) #len(train_alphabet))
    sentence_model = Classifier(order=args.ngram_length, alphabet_size=256) #len(train_alphabet))
    for i in train_instances:
        counts = {}
        for tok in i["tokens"]:
            word_model.train(tok["language"], tok["form"])
            counts[tok["language"]] = counts.get(tok["language"], 0) + 1
        maj_lang = sorted([(v, k) for k, v in counts.items()])[-1][1]
        sentence_model.train(maj_lang, " ".join([t["form"] for t in i["tokens"]]))
    return pickle.dumps((word_model, sentence_model))


def apply_model(model, test, args):
    word_model, sentence_model = pickle.loads(model)
    for i in range(len(test)):
        test[i]["scores"] = sentence_model.probabilities(" ".join([t["form"] for t in test[i]["tokens"]]))
        for j in range(len(test[i]["tokens"])):
            test[i]["tokens"][j]["scores"] = word_model.probabilities(test[i]["tokens"][j]["form"])
    return test
