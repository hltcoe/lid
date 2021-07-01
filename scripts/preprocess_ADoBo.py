import argparse
import json
import gzip
import re

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    with gzip.open(args.inputs[0], "rt") as ifd, gzip.open(args.output, "wt") as ofd:
        for i, sentence in enumerate(re.split(r"\n\s*\n", ifd.read()), 1):
            words = [re.match(r"^(.*)\t(\S+)$", w).groups() for w in sentence.strip().split("\n")]
            tokens = [{"form" : w, "language" : "spa" if l == "O" else l.lower().split("-")[-1]} for w, l in words]
            counts = {}
            for tok in tokens:
                counts[tok["language"]] = counts.get(tok["language"], 0) + 1
            total = sum(counts.values())
            ofd.write(
                json.dumps(
                    {
                        "id" : "ADoBo {}".format(i),
                        "tokens" : tokens,
                        "language" : {k : v / total for k, v in counts.items()},
                    }
                ) + "\n"
            )
