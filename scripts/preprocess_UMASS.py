import argparse
import json
import gzip
import zipfile
import csv
import io
import re

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    lengths = []
    labels = set()
    with zipfile.ZipFile(args.inputs[-1], mode="r") as zfd, gzip.open(args.output, "wt") as ofd:
        with zfd.open("umass_global_english_tweets-v1/all_annotated.tsv", mode="r") as ifd:
            for row in csv.DictReader(io.TextIOWrapper(ifd), delimiter="\t"):
                text = row["Tweet"]
                lengths.append(len(text))
                tokens = re.split(r"\s+", text)
                country = row["Country"]
                lengths.append(len(text))
                labels.add(country)
    print(len(labels), len(lengths), sum(lengths) / len(lengths))
