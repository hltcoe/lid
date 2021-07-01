import argparse
import json
import gzip
import tarfile
import re
import os.path
from iso639 import languages

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    counter = 0
    with gzip.open(args.output, "wt") as ofd, tarfile.open(args.inputs[0], "r") as tfd:
        first = True
        ofd.write("[\n")
        
        for fobj in tfd.getmembers():
            match = re.match(r"^appen/(.*)-10000_train.txt.gz$", fobj.name)
            if match != None:
                language = languages.get(alpha2=match.group(1)).part3
                with gzip.open(tfd.extractfile(fobj), "rt") as ifd:
                    for line in ifd:
                        counter += 1
                        tokens = re.split(r"\s+", line.strip())
                        doc = {
                            "id" : "Appen {}".format(counter),
                            "tokens" : [{"form" : tok, "language" : language} for tok in tokens],
                            "language" : {language : 1.0},
                        }
                        if first != True:
                            ofd.write(",\n")
                        first = False
                        ofd.write(json.dumps(doc))

        ofd.write("\n]")
