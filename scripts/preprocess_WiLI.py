import argparse
import json
import gzip
import zipfile
import io
import re
import os.path

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    with gzip.open(args.output, "wt") as ofd, zipfile.ZipFile(args.inputs[-1], "r") as zfd, zfd.open("x_train.txt") as xfd, zfd.open("y_train.txt") as yfd:
        for i, (x, y) in enumerate(zip(io.TextIOWrapper(xfd), io.TextIOWrapper(yfd)), 1):
            language = y.strip()
            tokens = [{"form" : w, "language" : language} for w in re.split(r"\s+", x.strip())]
            ofd.write(
                json.dumps(
                    {
                        "id" : "WiLI train {}".format(i),
                        "tokens" : tokens,
                        "language" : {language : 1.0},
                    }
                ) + "\n"
            )

