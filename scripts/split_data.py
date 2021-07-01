import gzip
import argparse
import random

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--train", dest="train", nargs=2, help="Train proportion and output file")
    parser.add_argument("--dev", dest="dev", nargs=2, help="Dev proportion and output file")
    parser.add_argument("--test", dest="test", nargs=2, help="Test proportion and output file")
    parser.add_argument("--seed", dest="seed", type=int, help="Random seed")
    args = parser.parse_args()

    train = float(args.train[0])
    dev = train + float(args.dev[0])
    test = dev + float(args.test[0])
    
    with gzip.open(args.input, "rt") as ifd, gzip.open(args.train[1], "wt") as train_ofd, gzip.open(args.dev[1], "wt") as dev_ofd, gzip.open(args.test[1], "wt") as test_ofd:
        ofds = {"train" : train_ofd, "dev" : dev_ofd, "test" : test_ofd}
        for ofd in ofds.values():
            ofd.write("[\n")
        first = {"train" : True, "dev" : True, "test" : True}
        for line in ifd:
            line = line.strip().rstrip(",")
            if line.strip() in ["[", "]"]:
                continue
            r = random.uniform(0.0, test)
            split = "train" if r < train else "dev" if r < dev else "test"
            if first[split] != True:
                ofds[split].write(",\n")
            first[split] = False
            ofds[split].write(line)
        for ofd in ofds.values():
            ofd.write("\n]")
