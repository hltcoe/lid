import gzip
import argparse
import random
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    with gzip.open(args.output, "wt") as ofd:
        ofd.write("[\n")
        first = True
        for fname in args.inputs:
            with gzip.open(fname, "rt") as ifd:
                for line in ifd:
                    if line.strip() not in ["[", "]"]:
                        if not first:
                            ofd.write(",\n")
                        first = False
                        ofd.write(line.strip().rstrip(","))
        ofd.write("\n]")
