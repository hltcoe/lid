import gzip
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("--sentence_heatmap_output", dest="sentence_heatmap_output", help="Output file")
    parser.add_argument("--token_heatmap_output", dest="token_heatmap_output", help="Output file")
    parser.add_argument("--sentence_bylength_output", dest="sentence_bylength_output", help="Output file")
    parser.add_argument("--token_bylength_output", dest="token_bylength_output", help="Output file")
    args, rest = parser.parse_known_args()

    token_gold_to_guess = {}
    sentence_gold_to_guess = {}
    sentence_by_length = {}
    token_by_length = {}
    languages = set()


    for input_fname in args.inputs:
        with gzip.open(input_fname, "rt") as ifd:
            for item in json.loads(ifd.read()):
                _, sentence_gold = max([(v, k) for k, v in item["language"].items()])
                if "scores" not in item:
                    item["scores"] = {}
                    for token in item["tokens"]:
                        for k, v in token["scores"].items():
                            item["scores"][k] = item["scores"].get(k, 0.0) + v
                _, sentence_guess = max([(v, k) for k, v in item["scores"].items()])
                slength = sum([len(t["form"]) for t in item["tokens"]])
                sentence_by_length[slength] = sentence_by_length.get(slength, []) + [sentence_gold == sentence_guess]
                sentence_gold_to_guess[sentence_gold] = sentence_gold_to_guess.get(
                    sentence_gold,
                    {}
                )
                sentence_gold_to_guess[sentence_gold][sentence_guess] = sentence_gold_to_guess[sentence_gold].get(sentence_guess, 0) + 1
                for token in item["tokens"]:
                    tlength = len(token["form"])
                    token_gold = token["language"]
                    languages.add(token_gold)
                    _, token_guess = max([(v, k) for k, v in token["scores"].items()])
                    token_by_length[tlength] = token_by_length.get(tlength, []) + [token_gold == token_guess]
                    languages.add(token_guess)
                    token_gold_to_guess[token_gold] = token_gold_to_guess.get(
                        token_gold,
                        {}
                    )
                    token_gold_to_guess[token_gold][token_guess] = token_gold_to_guess[token_gold].get(token_guess, 0) + 1
    languages = sorted(languages)

    #print(token_gold_to_guess)

    token_grid = numpy.zeros(shape=(len(languages), len(languages)))
    for i, gold in enumerate(languages):
        for j, guess in enumerate(languages):
            token_grid[i, j] = token_gold_to_guess.get(gold, {}).get(guess, 0.0)

    sentence_grid = numpy.zeros(shape=(len(languages), len(languages)))
    for i, gold in enumerate(languages):
        for j, guess in enumerate(languages):
            sentence_grid[i, j] = sentence_gold_to_guess.get(gold, {}).get(guess, 0.0)

    fs=4
    fig = matplotlib.figure.Figure(dpi=300)
    ax = fig.add_subplot()
    #fig, ax = plt.subplots()
    im = ax.imshow(sentence_grid)
    ax.set_xticklabels(languages)
    ax.set_yticklabels(languages)
    ax.set_xticks(numpy.arange(len(languages)))
    ax.set_yticks(numpy.arange(len(languages)))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", fontsize=fs)
    plt.setp(ax.get_yticklabels(), fontsize=fs)
    fig.tight_layout()
    fig.savefig(args.sentence_heatmap_output, bbox_inches="tight")


    fig = matplotlib.figure.Figure(dpi=300)
    ax = fig.add_subplot()
    #    fig, ax = plt.subplots()
    im = ax.imshow(token_grid)
    ax.set_xticklabels(languages)
    ax.set_yticklabels(languages)
    ax.set_xticks(numpy.arange(len(languages)))
    ax.set_yticks(numpy.arange(len(languages)))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", fontsize=fs)
    plt.setp(ax.get_yticklabels(), fontsize=fs)
    fig.tight_layout()
    fig.savefig(args.token_heatmap_output, bbox_inches="tight")

    binsize = 10
    _max = max(sentence_by_length.keys())
    _min = min(sentence_by_length.keys())
    _len = _max - _min + 1
    _blen = _len // binsize
    _blen = _blen + 1 if _blen * binsize < _len else _blen
    num = numpy.zeros(shape=(_len,))
    denom = numpy.zeros(shape=(_len,))
    bnum = numpy.zeros(shape=(_blen,))
    bdenom = numpy.zeros(shape=(_blen,))
    for k, v in sentence_by_length.items():
        num[k - _min] = len([x for x in v if x == True])
        denom[k - _min] = len(v)
    for i in range(_blen):
        bnum[i] = num[i * binsize : (i + 1) * binsize].sum()
        bdenom[i] = denom[i * binsize : (i + 1) * binsize].sum()
    accs = bnum / bdenom
    fig = matplotlib.figure.Figure(dpi=300)
    ax = fig.add_subplot()
    x = numpy.arange(len(bdenom))
    width = 0.9
    ax.bar(x - (width / 2), accs, width)
    ax.set_ylim(ymin=0.6)
    ax.set_xticks([i - 0.5 for i in range(_blen)])
    ax.set_xticklabels(["{}-{}".format(i * binsize, (i+1) * binsize) for i in range(_blen)])
    [x.set_rotation(30) for x in ax.get_xticklabels()]
    fig.tight_layout()
    fig.savefig(args.sentence_bylength_output, bbox_inches="tight")

    binsize = 10
    _max = max(token_by_length.keys())
    _min = min(token_by_length.keys())
    _len = _max - _min + 1
    _blen = _len // binsize
    _blen = _blen + 1 if _blen * binsize < _len else _blen
    num = numpy.zeros(shape=(_len,))
    denom = numpy.zeros(shape=(_len,))
    bnum = numpy.zeros(shape=(_blen,))
    bdenom = numpy.zeros(shape=(_blen,))
    for k, v in token_by_length.items():
        num[k - _min] = len([x for x in v if x == True])
        denom[k - _min] = len(v)
    for i in range(_blen):
        bnum[i] = num[i * binsize : (i + 1) * binsize].sum()
        bdenom[i] = denom[i * binsize : (i + 1) * binsize].sum()
    accs = bnum / bdenom
    fig = matplotlib.figure.Figure(dpi=300)
    ax = fig.add_subplot()
    x = numpy.arange(len(bdenom))
    width = 0.9
    ax.bar(x - (width / 2), accs, width)
    ax.set_ylim(ymin=0.6)
    ax.set_xticks([i - 0.5 for i in range(_blen)])
    ax.set_xticklabels(["{}-{}".format(i * binsize, (i+1) * binsize) for i in range(_blen)])
    [x.set_rotation(30) for x in ax.get_xticklabels()]
    #ax.set_xticks([i for i in range(_blen)])
    #ax.set_xticklabels(["{}-{}".format(i * binsize, (i+1) * binsize) for i in range(_blen)])
    fig.tight_layout()
    fig.savefig(args.token_bylength_output, bbox_inches="tight")
