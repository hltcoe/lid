import gzip
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    #parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("--sentence_output", dest="sentence_output", help="Output file")
    parser.add_argument("--token_output", dest="token_output", help="Output file")
    args, rest = parser.parse_known_args()

    token_gold_to_guess = {}
    sentence_gold_to_guess = {}
    languages = set()

    with gzip.open(args.input, "rt") as ifd:
        for item in json.loads(ifd.read()):
            _, sentence_gold = max([(v, k) for k, v in item["language"].items()])
            _, sentence_guess = max([(v, k) for k, v in item["scores"].items()])
            sentence_gold_to_guess[sentence_gold] = sentence_gold_to_guess.get(
                sentence_gold,
                {}
            )
            sentence_gold_to_guess[sentence_gold][sentence_guess] = sentence_gold_to_guess[sentence_gold].get(sentence_guess, 0) + 1
            for token in item["tokens"]:
                token_gold = token["language"]
                languages.add(token_gold)
                _, token_guess = max([(v, k) for k, v in token["scores"].items()])
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

    fig, ax = plt.subplots()
    im = ax.imshow(sentence_grid)
    ax.set_xticklabels(languages)
    ax.set_yticklabels(languages)

    # We want to show all ticks...
    ax.set_xticks(numpy.arange(len(languages)))
    ax.set_yticks(numpy.arange(len(languages)))
    ## ... and label them with the respective list entries
    #ax.set_xticklabels(farmers)
    #ax.set_yticklabels(vegetables)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    #for i in range(len(vegetables)):
    #    for j in range(len(farmers)):
    #        text = ax.text(j, i, harvest[i, j],
    #                       ha="center", va="center", color="w")

    #ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    #print(sentence_grid)
    fig.savefig(args.sentence_output)


    fig, ax = plt.subplots()
    im = ax.imshow(token_grid)
    ax.set_xticklabels(languages)
    ax.set_yticklabels(languages)

    # We want to show all ticks...
    ax.set_xticks(numpy.arange(len(languages)))
    ax.set_yticks(numpy.arange(len(languages)))
    ## ... and label them with the respective list entries
    #ax.set_xticklabels(farmers)
    #ax.set_yticklabels(vegetables)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    #for i in range(len(vegetables)):
    #    for j in range(len(farmers)):
    #        text = ax.text(j, i, harvest[i, j],
    #                       ha="center", va="center", color="w")

    #ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    #print(sentence_grid)
    fig.savefig(args.token_output)
