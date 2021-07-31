import json
import math
import logging

def beta(c_miss, c_fa, p_target):
    retval = (c_fa * (1 - p_target)) / (c_miss * p_target)
    logging.info(
        "Computed beta with p_target={:.2f}/c_miss={:.2f}/c_fa={:.2f} as: {:.2f}".format(
            p_target,
            c_miss,
            c_fa,
            retval
        )
    )
    return retval

def cavg(beta, p_miss, p_fa, labels):
    n_l = len(labels)
    total_miss = sum(p_miss.values())
    items = [v for v in p_fa.values()]
    total_fa = (beta * sum(sum([list(v.values()) for v in p_fa.values()], []))) / (len(labels) - 1)
    retval = (total_miss + total_fa) / len(labels)
    logging.info("Computed C_avg with beta={:.3f} to be: {:.3f}".format(beta, retval))
    if retval > 1.0 or retval < 0.0:
        print(beta, p_miss, p_fa, labels)
        raise Exception("C_avg of {} is out of bounds".format(retval))    
    return retval

def cprim(p_miss, p_fa, labels):
    b_1 = beta(c_miss=1.0, c_fa=1.0, p_target=0.5)
    b_2 = beta(c_miss=1.0, c_fa=1.0, p_target=0.1)
    cavg_1 = cavg(b_1, p_miss, p_fa, labels)
    cavg_2 = cavg(b_2, p_miss, p_fa, labels)
    retval = (cavg_1 + cavg_2) / 2.0
    logging.info(
        "Computed C_primary with cavg_1={:.3f}/cavg_2={:.3f} to be: {:.3f}".format(
            cavg_1,
            cavg_2,
            retval
        )
    )
    return retval

def c_metric(scores_list, gold_list):
    p_miss = {}
    p_fa = {}
    labels = set(["unk"])
    for scores, gold in zip(scores_list, gold_list):
        labels.add(gold)
        max_score = max(scores.values())
        unnorm_probs = {k : math.exp(v - max_score) for k, v in scores.items()}
        normalizer = sum(unnorm_probs.values())
        norm_probs = {k : v / normalizer for k, v in unnorm_probs.items()}
        p_miss[gold] = p_miss.get(gold, [])
        p_miss[gold].append(1.0 - norm_probs.get(gold, 0.0))
        p_fa[gold] = p_fa.get(gold, {})
        for label, prob in norm_probs.items():
            labels.add(label)
            if label != gold:
                p_fa[gold][label] = p_fa[gold].get(label, [])
                p_fa[gold][label].append(prob)
    p_miss = {k : sum(v) / len(v) for k, v in p_miss.items()}
    if "unk" not in p_miss:
        p_miss["unk"] = 0.0
    p_fa = {k : {l : sum(w) / len(gold_list) for l, w in v.items()} for k, v in p_fa.items()}
    retval = cprim(p_miss, p_fa, labels)
    if retval > 1.0 or retval < 0.0:
        raise Exception("C_metric of {} is out of bounds".format(retval))
    return retval

if __name__ == "__main__":
    pass
