import gzip
import json
from sklearn.metrics import f1_score
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args, rest = parser.parse_known_args()

    results = []
    exps = len(args.inputs) // 4
    for i in range(exps):
        print(i)
        config_f, output_f, tstats_f, astats_f = args.inputs[i * 4 : (i + 1) * 4]
        with gzip.open(config_f, "rt") as cfd, gzip.open(tstats_f, "rt") as tfd, gzip.open(astats_f, "rt") as afd:
            config = json.loads(cfd.read())
            tstats = json.loads(tfd.read())
            astats = json.loads(afd.read())
        labels = set()
        count = 0
        with gzip.open(output_f, "rt") as ifd:
            for item in json.loads(ifd.read()):
                labels.add(item["label"])
                for k in item["scores"].keys():
                    labels.add(k)
                count += 1
        guesses = []
        golds = []
        weights = []
        with gzip.open(output_f, "rt") as ifd:            
            for item in json.loads(ifd.read()):
                label = item["label"]                
                scores = item["scores"]
                if len(scores) > 0:
                    best = max([(v, k) for k, v in scores.items()])[1]
                else:
                    best = "unknown"
                total = sum(scores.values())
                golds.append(label)
                guesses.append(best)
                #print(len(labels), len(scores), label, best, output_f)
                if label in scores:
                    weights.append(scores[label])
                else:
                    weights.append((1.0 - total) / (len(labels) - len(scores)))
        score = f1_score(golds, guesses, average="macro")
        calib = sum(weights) / len(weights)
        results.append(config)
        results[-1]["f_score"] = score
        results[-1]["calibration"] = calib
        results[-1]["train_seconds"] = tstats["time"]
        results[-1]["apply_per_second"] = count / tstats["time"]
        for x in ["use_gpu", "hotspot_path"]:
            if x in results[-1]:
                del results[-1][x]

    with gzip.open(args.output, "wt") as ofd:
        ofd.write(json.dumps(results, indent=4))
