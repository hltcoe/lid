import argparse
import json
import gzip
import re
from iso639 import languages

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    with gzip.open(args.inputs[0], "rt") as ifd, gzip.open(args.output, "wt") as ofd:
        first = True
        ofd.write("[\n")
        for line in ifd:
            cid, spec, text = re.match(r"^tweets/(\d+)\t(\S+)\t(.*)", line.strip()).groups()
            code2, other = re.match(r"([a-z]+)(?:\-(\S+))?", spec).groups()            
            country = other if other in ["CN", "TW"] else None
            script = other if other in ["Latn"] else None
            transliteration = None
            language = code2 if code2 in languages.part3 else languages.get(alpha2=code2).part3
            properties = {"language" : language}
            if country != None:
                properties["country"] = country
            if script != None:
                properties["script"] = script
            if transliteration != None:
                properties["transliteration"] = transliteration
            tokens = []
            for t in re.split(r"\s+", text):
                tok = {"form" : t, "language" : language}
                tok.update(properties)
                tokens.append(tok)
            doc = {
                "id" : "Twitter {}".format(cid),
                "tokens" : tokens,
                "language" : {language : 1.0},
            }
            if first != True:
                ofd.write(",\n")
            first = False
            ofd.write(json.dumps(doc))
        ofd.write("\n]")
