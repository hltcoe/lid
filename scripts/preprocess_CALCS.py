import argparse
import json
import gzip
import zipfile
import io
import re
import os.path

comp = {
    "spaeng" : ["eng", "spa"],
    "hineng" : ["eng", "hin"],
    "msaea" : ["arz", "arb"],
    "nepeng" : ["eng", "nep"],
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    with gzip.open(args.output, "wt") as ofd:
        for fname in args.inputs:
            blob = re.match(r"lid_(.*)\.zip", os.path.basename(fname)).group(1)
            lang1, lang2 = comp[blob]
            with zipfile.ZipFile(fname, "r") as zfd, zfd.open("lid_{}/train.conll".format(blob), "r") as ifd:
                for i, sent in enumerate(re.split(r"\n\s*\n", io.TextIOWrapper(ifd).read()), 1):
                    tokens = [{"form" : w,
                               "language" : lang1 if l == "lang1" else lang2 if l == "lang2" else "unk"
                               } for w, l in [line.rstrip().split("\t") for line in sent.strip().split("\n")[1:]] if w.strip() != ""]
                    ofd.write(
                        json.dumps(
                            {
                                "id" : "CALCS {} train {}".format(blob, i),
                                "tokens" : tokens
                            }
                        ) + "\n"
                    )

