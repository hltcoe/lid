import math
import gzip
import json
from sklearn.metrics import f1_score, det_curve, plot_det_curve, plot_roc_curve, brier_score_loss
import argparse

def beta(p_target, c_miss=1.0, c_fa=1.0):
    retval = (c_fa * (1 - p_target)) / (c_miss * p_target)
    print(
        "Computing beta for target probability {:.1f}/cost of miss {:.1f}/cost of false alarm {:.1f} as: {:.2f}".format(
            p_target,
            c_miss,
            c_fa,
            retval
        )
    )
    return retval

#def cavg(beta, target_miss_probs, target_as_non_target_probs):
def cavg(beta, p_miss, p_fa):
    total_miss = sum(p_miss.values())
    items = [v for v in p_fa.values()]
    total_fa = (beta * sum(sum([list(v.values()) for v in p_fa.values()], []))) / (len(p_miss) - 1)
    retval = (total_miss + total_fa) / len(p_miss)
    print("Computing C_avg with beta {:.3f} as: {:.3f}".format(beta, retval))
    return retval

#def cprim(target_miss_probs, target_as_non_target_probs):
def cprim(p_miss, p_fa):
    retval = (cavg(beta(0.5), p_miss, p_fa) + cavg(beta(0.1), p_miss, p_fa)) / 2.0
    print("Computing C_primary as: {:.3f}".format(retval))
    return retval

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args, rest = parser.parse_known_args()

    results = []
    exps = len(args.inputs) // 4
    for i in range(exps):
        config_f, output_f, tstats_f, astats_f = args.inputs[i * 4 : (i + 1) * 4]
        with gzip.open(config_f, "rt") as cfd, gzip.open(tstats_f, "rt") as tfd, gzip.open(astats_f, "rt") as afd:
            config = json.loads(cfd.read())
            tstats = json.loads(tfd.read())
            astats = json.loads(afd.read())
        languages = set()
        with gzip.open(output_f, "rt") as ifd:
            for item in json.loads(ifd.read()):
                for token in item["tokens"]:
                    languages.add(token["language"])
        count = 0
        p_miss = {k : [] for k in languages}
        p_fa = {k : {l : [] for l in languages if l != k} for k in languages}
        token_pairs = []
        sentence_pairs = []

        with gzip.open(output_f, "rt") as ifd:
            for item in json.loads(ifd.read()):
                pairs = []
                counts = {}
                for token in item["tokens"]:
                    token["scores"] = item["scores"]
                    guess = sorted([(v, k) for k, v in token["scores"].items()])[-1][1]                    
                    counts[guess] = counts.get(guess, 0) + 1
                    gold = token["language"]
                    pairs.append((guess, gold))
                    max_val = max(token["scores"].values())
                    probs = {k : math.exp(v - max_val) for k, v in token["scores"].items()}
                    total_prob = sum(probs.values())
                    correct = token["language"]
                    probs = {k : v / total_prob for k, v in probs.items()}
                    pm = 1.0 - probs.get(correct, 0.0)
                    p_miss[correct] = p_miss.get(correct, [])
                    p_miss[correct].append(pm)
                    for other_lang in languages:
                        if other_lang != correct:
                            prob = probs.get(other_lang, 0.0)
                            p_fa[correct] = p_fa.get(correct, {})
                            p_fa[correct][other_lang] = p_fa[correct].get(other_lang, [])
                            p_fa[correct][other_lang].append(prob)
                token_pairs.append(pairs)
                guess = sorted([(v, k) for k, v in item["scores"].items()])[-1][1]
                gold = sorted([(v, k) for k, v in counts.items()])[-1][1]
                #if guess != gold:
                #    print(gold)
                sentence_pairs.append((guess, gold))
        token_pairs = sum(token_pairs, [])
        token_score = f1_score([guess for guess, gold in token_pairs], [gold for guess, gold in token_pairs], average="macro")
        print(len(sentence_pairs))
        sentence_score = f1_score([guess for guess, gold in sentence_pairs], [gold for guess, gold in sentence_pairs], average="macro")
        # calib = sum(weights) / len(weights)
        #print(p_miss)
        p_miss = {k : sum(v) / len(v) for k, v in p_miss.items()}
        p_fa = {k : {l : sum(vv) / len(vv) for l, vv in v.items()} for k, v in p_fa.items()}
        #print(p_miss)
        #print(p_fa)
        #sys.exit()
        results.append(config)
        results[-1]["c_primary"] = cprim(p_miss, p_fa)
        results[-1]["token_f_score"] = token_score
        results[-1]["sentence_f_score"] = sentence_score        
        #results[-1]["calibration"] = calib
        results[-1]["train_seconds"] = tstats["time"]
        results[-1]["apply_per_second"] = count / tstats["time"]
        for x in ["use_gpu", "hotspot_path"]:
            if x in results[-1]:
                del results[-1][x]
    
    with gzip.open(args.output, "wt") as ofd:
        ofd.write(json.dumps(results, indent=4))
# # prepare plots
# fig, [ax_roc, ax_det] = plt.subplots(1, 2, figsize=(11, 5))

# for name, clf in classifiers.items():
#     clf.fit(X_train, y_train)

#     plot_roc_curve(clf, X_test, y_test, ax=ax_roc, name=name)
#     plot_det_curve(clf, X_test, y_test, ax=ax_det, name=name)
