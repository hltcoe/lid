import argparse
import gzip
import json
import pandas
import re
import os.path
from glob import glob

preamble = """
\\documentclass{article}
\\usepackage[a4paper,margin=1cm]{geometry}
\\usepackage{color}
\\usepackage{graphicx}
\\usepackage{booktabs}
\\usepackage{caption}
\\usepackage{subcaption}
\\title{Language Identification for Text}
\\author{Tom Lippincott}
\\begin{document}
\\maketitle




"""

postamble = """
\\nocite{*}
\\bibliographystyle{plain}
\\bibliography{report}
\\end{document}
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--data_summary", dest="data_summary", help="")
    parser.add_argument("--correlations", dest="correlations", help="")
    parser.add_argument("--output", dest="output", help="Output path")
    #parser.add_argument(dest="figures", nargs="+", help="")
    args, rest = parser.parse_known_args()

    
    with open(args.data_summary, "rt") as ifd:
        data_summary = "\\section*{Datasets}\n" + ifd.read()

    figures = {}
    for fname in glob(os.path.join(args.output, "*png")):
        try:
            level, dname, mname, vtype = re.match(r"^(\w+)_(\w+)_(\w+)_(\w+).png", os.path.basename(fname)).groups()
        except:
            level, dname, vtype = re.match(r"^(\w+)_(\w+)_(\w+).png", os.path.basename(fname)).groups()
            mname = "all"
        figures[dname] = figures.get(dname, {})
        figures[dname][level] = figures[dname].get(level, {})
        figures[dname][level][vtype] = figures[dname][level].get(vtype, {})
        figures[dname][level][vtype][mname] = os.path.basename(fname)

    labels = set()
    sentence_summed_counts, word_summed_counts, by_sentence_length, by_word_length, experiments = {}, {}, {}, {}, {}
    tokens_per_second = {}
    with gzip.open(args.input, "rt") as ifd:
        for exp in json.loads(ifd.read()):
            mn = exp["MODEL_NAME"]
            tokens_per_second[mn] = tokens_per_second.get(mn, []) + [exp["apply_tokens_per_second"]]
            for k, v in [x for x in exp.items() if x[0] not in ["sentence_gold_to_guess", "token_gold_to_guess", "ngram_length", "ngram_path", "batch_size", "dev_interval", "training_observations"]]:
                experiments[k] = experiments.get(k, [])
                experiments[k].append(v)
    funcs = ["mean"] # "std"]
    experiments = pandas.DataFrame(experiments)
    table = "\\newpage\n\\section*{Performance}\n" + "\\begin{table}[h]\n\\centering\n" + experiments.groupby(["DATASET_NAME", "MODEL_NAME"]).agg(
        {
            "token_f_score" : funcs,
            "sentence_f_score" : funcs,
            "token_c_primary" : funcs,
            "sentence_c_primary" : funcs,
        }
    ).to_latex(float_format="%.3f") + """
  \\caption{F-score is equally weighted across languages.  Cavg is the evaluation metric defined for the 2017 NIST language recognition evaluation \cite{lre}.}
\\end{table}
"""
    table = "\n".join([l if "token\\_f\\_score" not in l else "Dataset & Model & \\multicolumn{2}{c}{F-Score} & \\multicolumn{2}{c}{Cavg} \\\\\n& & Token & Sentence & Token & Sentence \\\\" for l in table.split("\n") if "mean" not in l and "DATASET\\_NAME" not in l]).replace("ngram", "VaLID")
    sizes = {
        "ngram" : 8.5,
        "HOTSPOT" : 2.8,
        "Hierarchical" : 72
    }
    model_section = """
\\begin{table}[h]
  \\begin{tabular}{lrr}
  \\toprule
  Model & Size & Speed \\\\
  \\midrule
  %s
  \\bottomrule
  \\end{tabular}
    \\caption{Size is in bytes serialized on disk for the Twitter model, speed is in tokens processed per second across all experiments.  The Hierarchical model is run on GPU.}
\\end{table}
"""%("\n".join(["    {} & {:.1f}m & {} \\\\".format("VaLID" if k == "ngram" else k, sizes[k], int(sum(v) / len(v))) for k, v in tokens_per_second.items()]))

    vis = ""
    with open(args.correlations, "rt") as ifd:
        corr = ifd.read()
        #vis += corr
    for dname, levels in figures.items():
        vis += """
\\clearpage
        """
        for level, vtypes in levels.items():
            lvl = "Sentence" if level == "sentence" else "Token"
            for vtype, models in vtypes.items():
                caption = ("%s-level confusion matrices on %s" if vtype == "hm" else "%s-level accuracy by length on %s")%(lvl, dname)
                model_figs = ["""\\begin{subfigure}[b]{%.3f\\textheight}
  \\includegraphics[width=\\textwidth]{%s}
  \\caption{%s}
\\end{subfigure}\n\\\\""" % (0.3 if vtype == "hm" else 0.5, f, "VaLID" if n == "ngram" else n) for n, f in sorted(list(models.items()))]
                vis += "\n\\begin{figure}\n\\centering\n" + "\n".join(model_figs) + "\n\\caption{%s}\n\\end{figure}"%(caption)

    with open(os.path.join(args.output, "data_table.tex"), "wt") as ofd:
        ofd.write(data_summary) 
    with open(os.path.join(args.output, "figures.tex"), "wt") as ofd:
        ofd.write(vis)
    with open(os.path.join(args.output, "correlations.tex"), "wt") as ofd:
        ofd.write(corr)
    with open(os.path.join(args.output, "models.tex"), "wt") as ofd:
        ofd.write(model_section)
    with open(os.path.join(args.output, "performance.tex"), "wt") as ofd:
        ofd.write(table)
        
        #ofd.write("\n".join([preamble, data_summary, model_section, table, vis, postamble]))
