import math
import gzip
import json
from sklearn.metrics import f1_score, det_curve, plot_det_curve, plot_roc_curve, brier_score_loss, top_k_accuracy_score
from calibration import c_metric
import logging
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args, rest = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)

    results = []
    exps = len(args.inputs) // 4
    for i in range(exps):
        config_f, output_f, tstats_f, astats_f = args.inputs[i * 4 : (i + 1) * 4]
        print(config_f)
        with gzip.open(config_f, "rt") as cfd, gzip.open(tstats_f, "rt") as tfd, gzip.open(astats_f, "rt") as afd:
            config = json.loads(cfd.read())
            tstats = json.loads(tfd.read())
            astats = json.loads(afd.read())
        languages = set()
        token_scores, token_golds = [], []
        sentence_scores, sentence_golds = [], []
        sentence_gold_to_guess, token_gold_to_guess = {}, {}
        by_sentence_length, by_token_length = {}, {}
        with gzip.open(output_f, "rt") as ifd:
            for item in json.loads(ifd.read()):

                if "scores" in item:
                    _sentence_scores = item["scores"]
                else:
                    _sentence_scores = {}
                    for token in item["tokens"]:
                        for k, v in token["scores"].items():
                            _sentence_scores[k] = _sentence_scores.get(k, 0.0) + v
                _, sentence_gold = max([(v, k) for k, v in item["language"].items()])
                _, sentence_guess = max([(v, k) for k, v in _sentence_scores.items()]) #item["scores"].items()])
                by_sentence_length[len(item["tokens"])] = by_sentence_length.get(len(item["tokens"]), [])
                by_sentence_length[len(item["tokens"])].append(1 if sentence_gold == sentence_guess else 0)
                sentence_gold_to_guess[sentence_gold] = sentence_gold_to_guess.get(sentence_gold, {})
                sentence_gold_to_guess[sentence_gold][sentence_guess] = sentence_gold_to_guess[sentence_gold].get(sentence_guess, 0) + 1
                sentence_scores.append(_sentence_scores) #item["scores"])
                sentence_golds.append(sentence_gold)
                for token in item["tokens"]:
                    token_gold = token["language"]
                    languages.add(token_gold)
                    token_scores.append(token["scores"])
                    token_golds.append(token_gold)
                    _, token_guess = max([(v, k) for k, v in token["scores"].items()])
                    by_token_length[len(item["tokens"])] = by_token_length.get(len(item["tokens"]), [])
                    by_token_length[len(item["tokens"])].append(1 if token_gold == token_guess else 0)
                    token_gold_to_guess[token_gold] = token_gold_to_guess.get(token_gold, {})
                    token_gold_to_guess[token_gold][token_guess] = token_gold_to_guess[token_gold].get(token_guess, 0) + 1

                    languages.add(token_guess)
        languages = list(languages)
        config["sentence_gold_to_guess"] = sentence_gold_to_guess
        config["token_gold_to_guess"] = token_gold_to_guess
        config["by_sentence_length"] = by_sentence_length
        config["by_token_length"] = by_token_length

        config["token_c_primary"] = c_metric(token_scores, token_golds)
        config["sentence_c_primary"] = c_metric(sentence_scores, sentence_golds)
        #config["token_top_3_accuracy"] = top_k_accuracy_score(
        #    [[scores.get(k) for k in languages] for scores in token_scores],
        #    [languages.index(k) for k in token_golds],
        #    k=3
        #)
        #config["sentence_top_3_accuracy"] = top_k_accuracy_score(sentence_scores, sentence_golds, k=3)
        config["token_f_score"] = f1_score(
            [max([(v, k) for k, v in s.items()])[1] for s in token_scores],
            token_golds,
            average="macro"
        )
        config["sentence_f_score"] = f1_score(
            [max([(v, k) for k, v in s.items()])[1] for s in sentence_scores],
            sentence_golds,
            average="macro"
        )
        config["train_seconds"] = tstats["time"]
        config["apply_tokens_per_second"] = (len(token_golds) * 2) / astats["time"]
        for x in ["use_gpu", "hotspot_path"]:
            if x in config:
                del config[x]                
        results.append(config)
    
    with gzip.open(args.output, "wt") as ofd:
        ofd.write(json.dumps(results, indent=4))
