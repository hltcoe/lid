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
        ofd.write("""\\begin{table}[h]
  \\begin{tabular}{lrrrrrp{.5\\textwidth}}
    \\toprule
    Name & Langs & Docs & \\multicolumn{2}{c}{Avg char length}  & \\%-Switched & Description \\\\
    & & & Document & Token & & \\\\
    \\midrule
""")
        for fname in args.inputs:
            name = os.path.splitext(os.path.splitext(os.path.basename(fname))[0])[0]
            languages = {}
            sentence_lengths = []
            token_lengths = []
            code_sw = []
            with gzip.open(fname, "rt") as ifd:
                for line in ifd:
                    line = line.strip().rstrip(",")
                    if line not in ["[", "]"]:
                        j = json.loads(line)
                        sentence_length = 0
                        prev = None
                        cs = False
                        for tok in j["tokens"]:
                            sentence_length += (1 + len(tok["form"]))
                            token_lengths.append(len(tok["form"]))
                            if prev != None and prev != tok["language"]:
                                cs = True
                            prev = tok["language"]
                            languages[tok["language"]] = languages.get(tok["language"], 0) + 1
                        code_sw.append(1 if cs else 0)
                        sentence_lengths.append(sentence_length)
            ofd.write("""    {} & {} & {} & {:.2f} & {:.2f} & {:.2f} & \\\\
""".format(name, len(languages), len(sentence_lengths), sum(sentence_lengths) / len(sentence_lengths), sum(token_lengths) / len(token_lengths), int(100 * sum(code_sw) / len(code_sw))))
        ofd.write("""    \\bottomrule
  \\end{tabular}
\\end{table}
""")
