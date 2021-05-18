import argparse
import json
import gzip
import tarfile
import csv
import io
import re

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    with tarfile.open(args.inputs[-1], "r") as tfd, tfd.extractfile("sentences_detailed.csv") as ifd, gzip.open(args.output, "wt") as ofd:
        first = True
        ofd.write("[\n")
        for row in csv.DictReader(io.TextIOWrapper(ifd), delimiter="\t", fieldnames=["id", "language", "text", "user", "one", "two"]):
            tokens = [{"form" : tok, "language" : row["language"]} for tok in re.split(r"\s+", row["text"])]
            if first != True:
                ofd.write(",\n")
            first = False
            ofd.write(json.dumps({"id" : "Tatoeba {}".format(row["id"]), "tokens" : tokens}))            
        ofd.write("\n]")
