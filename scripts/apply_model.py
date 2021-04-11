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
    parser.add_argument("--model_input", dest="model_input", help="Model file")
    parser.add_argument("--data_input", dest="data_input", help="Data to apply model to")
    parser.add_argument("--applied_output", dest="applied_output", help="Output file for applied results")
    parser.add_argument("--statistical_output", dest="statistical_output", help="Output file for application statistics")
    args, rest = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)
    
    start = time.time()
    module = importlib.import_module(args.model_name)

    with gzip.open(args.data_input, "rt") as ifd:
        test = json.loads(ifd.read())
    
    logging.info("Loading data...")
    instances = []
    for item in test:
        seq = []
        label_counts = {}
        for token in item["tokens"]:
            lang = token["language"]
            label_counts[lang] = label_counts.get(lang, 0)
            label_counts[lang] += 1
            seq.append(token["form"])
        label = max([(v,k) for k,v in label_counts.items()])[1]
        instances.append({"label" : label, "sequence" : " ".join(seq), "id" : item["id"]})


    with gzip.open(args.model_input, "rb") as ifd, gzip.open(args.applied_output, "wt") as ofd:
        ofd.write(json.dumps(module.apply_model(ifd.read(), instances, rest)))
    end = time.time()
    with gzip.open(args.statistical_output, "wt") as ofd:
        ofd.write(json.dumps({"time" : end - start}))
