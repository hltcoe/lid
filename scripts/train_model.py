import argparse
import json
import gzip
import importlib
import logging
import pickle
import time
import random

def limit_length(items, max_length):
    for i in range(len(items)):
        full_length = len(items[i]["tokens"])
        if full_length > max_length:
            start = random.randint(0, full_length - max_length)
            items[i]["tokens"] = items[i]["tokens"][start:start + max_length]
    return items

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", dest="model_name", help="Model name to train") 
    parser.add_argument("--train_input", dest="train_input", help="Training data")
    parser.add_argument("--dev_input", dest="dev_input", help="Dev data")
    parser.add_argument("--model_output", dest="model_output", help="Output file for model")
    parser.add_argument("--statistical_output", dest="statistical_output", help="Output file for training statistics")
    args, rest = parser.parse_known_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(name)s - %(asctime)s - %(levelname)s - %(message)s'
    )
    random.seed(0)

    start = time.time()
    module = importlib.import_module(args.model_name)
    
    logging.info("Loading training data...")
    with gzip.open(args.train_input, "rt") as ifd:
        train = json.loads(ifd.read())
    random.shuffle(train)
    train = limit_length(train[:50000], 20)

    logging.info("Loading dev data...")
    with gzip.open(args.dev_input, "rt") as ifd:
        dev = json.loads(ifd.read())

    random.shuffle(dev)

    dev = limit_length(dev[:5000], 20)
    
    model = module.train_model(train, dev, args.model_output, rest)

    end = time.time()
    with gzip.open(args.statistical_output, "wt") as ofd:
        ofd.write(json.dumps({"time" : end - start}))
    with gzip.open(args.model_output, "wb") as ofd:
        ofd.write(model)
