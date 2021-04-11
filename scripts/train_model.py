import argparse
import json
import gzip
import importlib
import logging
import pickle
import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", dest="model_name", help="Model name to train") 
    parser.add_argument("--train_input", dest="train_input", help="Training data")
    parser.add_argument("--dev_input", dest="dev_input", help="Dev data")
    parser.add_argument("--model_output", dest="model_output", help="Output file for model")
    parser.add_argument("--statistical_output", dest="statistical_output", help="Output file for training statistics")
    args, rest = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)
    
    start = time.time()
    module = importlib.import_module(args.model_name)
    with gzip.open(args.train_input, "rt") as ifd:
        train = json.loads(ifd.read())
    with gzip.open(args.dev_input, "rt") as ifd:
        dev = json.loads(ifd.read())
    
    logging.info("Loading training data...")
    train_instances = []
    for item in train:
        seq = []
        label_counts = {}
        for token in item["tokens"]:
            lang = token["language"]
            label_counts[lang] = label_counts.get(lang, 0)
            label_counts[lang] += 1
            seq.append(token["form"])
        label = max([(v,k) for k,v in label_counts.items()])[1]
        train_instances.append({"label" : label, "sequence" : " ".join(seq)})

    logging.info("Loading dev data...")
    dev_instances = []
    for item in dev:
        seq = []
        label_counts = {}
        for token in item["tokens"]:
            lang = token["language"]
            label_counts[lang] = label_counts.get(lang, 0)
            label_counts[lang] += 1
            seq.append(token["form"])
        label = max([(v,k) for k,v in label_counts.items()])[1]
        dev_instances.append({"label" : label, "sequence" : " ".join(seq)})




        
    model = module.train_model(train_instances, dev_instances, args.model_output, rest)
    end = time.time()
    with gzip.open(args.statistical_output, "wt") as ofd:
        ofd.write(json.dumps({"time" : end - start}))
    with gzip.open(args.model_output, "wb") as ofd:
        ofd.write(model)
