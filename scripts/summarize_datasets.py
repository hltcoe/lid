import argparse
import os.path
import json
import gzip

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    with open(args.output, "wt") as ofd:
        ofd.write("""\\begin{tabular}{llllp{.5\\textwidth}}
\\hline
Name & Languages & Documents & Avg length & Description \\\\
\\hline
""")
        for fname in args.inputs:
            name = os.path.splitext(os.path.basename(fname))[0]
            languages = {}
            lengths = []
            with gzip.open(fname, "rt") as ifd:
                for line in ifd:
                    line = line.strip().rstrip(",")
                    if line not in ["[", "]"]:
                        j = json.loads(line)
                        length = 0
                        for tok in j["tokens"]:
                            length += (1 + len(tok["form"]))
                            languages[tok["language"]] = languages.get(tok["language"], 0) + 1
                        lengths.append(length)
            ofd.write("""{} & {} & {} & {:.3f} & \\\\
""".format(name, len(languages), len(lengths), sum(lengths) / (1 + len(lengths))))
        ofd.write("""\\hline
\\end{tabular}
        """)
