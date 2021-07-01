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
    parser.add_argument("--model_input", dest="model_input", help="Model file")
    parser.add_argument("--data_input", dest="data_input", help="Data to apply model to")
    parser.add_argument("--applied_output", dest="applied_output", help="Output file for applied results")
    parser.add_argument("--statistical_output", dest="statistical_output", help="Output file for application statistics")
    args, rest = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)
    random.seed(0)    

    module = importlib.import_module(args.model_name)

    with gzip.open(args.data_input, "rt") as ifd:
        instances = json.loads(ifd.read())

    random.shuffle(instances)
    instances = limit_length(instances, 20)

    start = time.time()
    #instances = test

    #logging.info("Loading data...")
    #instances = []
    #for item in test:
    #    seq = []
    #    label_counts = {}
    #    for token in item["tokens"]:
    #        lang = token["language"]
    #        label_counts[lang] = label_counts.get(lang, 0)
    #        label_counts[lang] += 1
    #        seq.append(token["form"])
    #    label = max([(v,k) for k,v in label_counts.items()])[1]
    #    instances.append({"label" : label, "sequence" : " ".join(seq), "id" : item["id"]})
    with gzip.open(args.model_input, "rb") as ifd:
        out = module.apply_model(ifd.read(), instances, rest)
    end = time.time()
    if isinstance(out[0], (dict,)):
        realout = out
    else:
        realout = instances
        i = 0
        for tchunk, schunk in out[0]:
            for row, srow in zip(tchunk, schunk):
                realout[i]["scores"] = {str(k) : srow[v] for k, v in out[1].items()}
                #print(len(instances[i]["tokens"]))
                for j in range(len(realout[i]["tokens"])):
                    word = row[j]
                    realout[i]["tokens"][j]["scores"] = {str(k) : word[v] for k, v in out[1].items()}
                i += 1


    with gzip.open(args.applied_output, "wt") as ofd:
        ofd.write(json.dumps(realout, indent=4))

    with gzip.open(args.statistical_output, "wt") as ofd:
        ofd.write(json.dumps({"time" : end - start}, indent=4))
