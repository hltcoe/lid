import argparse
import gzip
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", help="Input file")
    parser.add_argument("--output", dest="output", help="Output file")
    args, rest = parser.parse_known_args()

    config = dict(json.loads(args.config))
    with gzip.open(args.output, "wt") as ofd:
        ofd.write(json.dumps(config))
