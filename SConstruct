import os
import os.path
import logging
import random
import subprocess
import shlex
import gzip
import re
import functools
import time
import imp
import sys
import json
from hashlib import md5
import steamroller

#
# Workaround needed to fix bug with SCons and the pickle module (maybe no longer necessary?)
#
del sys.modules['pickle']
sys.modules['pickle'] = imp.load_module('pickle', *imp.find_module('pickle'))
import pickle

#
# Variables are the crux of the build system.  These default values can be overridden by top-
# level declarations in a "custom.py" file (e.g. 'OUTPUT_WIDTH = 1000') or on the command
# line (e.g. 'scons OUTPUT_WIDTH=1000').
#
vars = Variables("custom.py")


vars.AddVariables(
    ("OUTPUT_WIDTH", "Limit progress-lines to a maximum number of characters.", 1000),
    ("BATCH_SIZE", "Self-explanatory.", 512),
    ("TRAIN_PROPORTION", "Proportion of data for training.", 0.8),
    ("DEV_PROPORTION", "Proportion of data for development/tuning.", 0.1),
    ("USE_GRID", "", False),
    ("USE_GPU", "", False),
    ("TEST_PROPORTION", "Proportion of data for testing.", 0.1),
    ("FOLDS", "How many full, randomized sets of experiments to run.", 1),
    ("HOTSPOT_PATH", "", "data/hotspot-8.17-SNAPSHOT-jar-with-dependencies.jar"),
    ("NGRAM_PATH", "", "/home/tom/projects/text-classification/ngram"),
    (
        "DATA_PATH",
        "A base path for e.g. the DATASETS below (should probably be set to somewhere in your home directory etc).",
        "data/"
    ),
    (
        "DATASETS",
        "This dictionary maps the name of a dataset to the locations of its canonical files (i.e. as-provided/downloaded).",
        {
            "Twitter" : ["${DATA_PATH}/twitter/balanced_lid.txt.gz"],
            "Appen" : ["${DATA_PATH}/appen/sms.tgz"],
            "CALCS" : ["${{DATA_PATH}}/calcs2021/lid_{}.zip".format(x) for x in ["hineng", "msaea", "nepeng", "spaeng"]],
            "ADoBo" : ["${DATA_PATH}/ADoBo/training.conll.gz"],
            #"WiLI" : ["${DATA_PATH}/wikipedia/wili-2018.zip"],
        }
    ),
    (
        "MODELS",
        "Each model name is associated with lists of hyper-parameters for n-ary experiments.",
        {
            "VaLID" : {
                "ngram_length" : 4,
                "can_use_gpu" : False,
            },
            #"HOTSPOT" : {
            #    "hotspot_path" : "${HOTSPOT_PATH}",
            #    "can_use_gpu" : False,
            #},
            "Hierarchical" : {
                "can_use_gpu" : True,
                "training_observations" : 5000000,
                "batch_size" : 8,
                "dev_interval" : 200000,
            },
        }
    ),
)

#
# Pass "steamroller.generate" to get access to grid-aware builders and some helper functions.
#
env = Environment(variables=vars, ENV=os.environ, TARFLAGS="-c -z", TARSUFFIX=".tgz",
                  tools=["default", steamroller.generate],
)

#
# Replace the default progress/dry-run command-printing function with one that uses the
# OUTPUT_WIDTH limiting variable.
#
def print_cmd_line(s, target, source, env):
    if len(s) > int(env["OUTPUT_WIDTH"]):
        print(s[:int(float(env["OUTPUT_WIDTH"]) / 2) - 2] + "..." + s[-int(float(env["OUTPUT_WIDTH"]) / 2) + 1:])
    else:
        print(s)
env['PRINT_CMD_LINE_FUNC'] = print_cmd_line

#
# Replace the default out-of-date decider with one (that seems to be a) bit more robust
# when run on a grid while producing the same results otherwise.
#
def decider(dependency, target, prev_ni, repo_node):    
    dep_stamp = dependency.get_timestamp()
    tgt_stamp = target.get_timestamp()
    if dep_stamp > tgt_stamp or not dependency.exists():
        return True
    return False
env.Decider(decider)

#
# These builders are completely independent of the dataset/model/hyper-parameters used before/after
# them, so nothing too fancy is needed, they just invoke a corresponding script with the appropriate
# flags.
#
env.AddBuilder(
    "SplitData",
    "scripts/split_data.py",
    "--input ${SOURCES[0]} --train ${TRAIN_PROPORTION} ${TARGETS[0]} --dev ${DEV_PROPORTION} ${TARGETS[1]} --test ${TEST_PROPORTION} ${TARGETS[2]}"
)
env.AddBuilder(
    "Evaluate",
    "scripts/evaluate.py",
    "${SOURCES} --output ${TARGETS[0]}",
    other_deps=["scripts/calibration.py"],
)
env.AddBuilder(
    "Figures",
    "scripts/figures.py",
    "${SOURCES[0]} --sentence_heatmap ${TARGETS[0]} --sentence_bylength ${TARGETS[1]} --token_heatmap ${TARGETS[2]} --token_bylength ${TARGETS[3]}"
)
env.AddBuilder(
    "SaveConfig",
    "scripts/save_config.py",
    "${SOURCES} --output ${TARGETS[0]} --config '${CONFIG}'"
)
env.AddBuilder(
    "SummarizeDatasets",
    "scripts/summarize_datasets.py",
    "${SOURCES} --output ${TARGETS[0]}"
)
env.AddBuilder(
    "CreateReport",
    "scripts/create_report.py",
    "--input ${SOURCES[0]} --output ${TARGETS[0]} --dataset_summary ${SOURCES[1]} ${SOURCES[2:]}"
)
env.AddBuilder(
    "CompileReport",
    "",
    "-interaction nonstopmode -jobname ${TARGETS[0].name[:-4]} ${SOURCES[0].abspath}",
    interpreter="/usr/bin/pdflatex",
    chdir="work"
)

#
# This for-loop adds a build rule for each dataset defined in the SCons environment variables, relying
# on there being a Python script "scripts/preprocess_${DATASET_NAME}.py" that accepts arguments of the form
# "INFILE1 INFILE2 ... --output OUTFILE" and turns the input files into a single output file with a
# simple, uniform JSON format.
#
for dataset_name, filenames in env["DATASETS"].items():
    rule_name = "Preprocess{}".format(dataset_name)
    builder = env.AddBuilder(
        rule_name,
        "scripts/preprocess_{}.py".format(dataset_name),
        "${' '.join([\"'%s'\" % (s) for s in SOURCES])} --output ${TARGETS[0]}",
    )

#
# Similarly to the datasets, the models to use are defined in the SCons environment variables, and
# this for-loop creates build rules for training and applying them.  It too relies on scripts
# ("scripts/${MODEL_NAME}.py", "scripts/train_model.py", "scripts/apply_model.py") existing that
# accept a few generic arguments ("--train_input", "--model_output", etc) and also model-specific
# arguments based on the model_spec ("--hidden_size", "--markov_depth", etc).
#
for model_name, model_spec in env["MODELS"].items():
    train_rule_name = "Train{}".format(model_name)
    apply_rule_name = "Apply{}".format(model_name)
    use_gpu=model_spec.get("can_use_gpu", False) and env["USE_GPU"]
    env.AddBuilder(
        train_rule_name,
        "scripts/train_model.py".format(model_name),
        " ".join(
            ["--train_input ${SOURCES[0]} --dev_input ${SOURCES[1]} --model_output ${TARGETS[0]} --statistical_output ${TARGETS[1]} --model_name ${MODEL_NAME}"] + ["--{} {}".format(k, v) for k, v in model_spec.items()] + ["--use_gpu" if use_gpu else ""]
        ),
        other_deps=["scripts/{}.py".format(model_name)],
        use_gpu=use_gpu
    )
    env.AddBuilder(
        apply_rule_name,
        "scripts/apply_model.py".format(model_name),
        " ".join(
            ["--model_name ${MODEL_NAME} --model_input ${SOURCES[0]} --data_input ${SOURCES[1]} --applied_output ${TARGETS[0]} --apply_batch_size 128 --statistical_output ${TARGETS[1]}"] + ["--{} {}".format(k, v) for k, v in model_spec.items()]
        ),
        other_deps=["scripts/{}.py".format(model_name)],
        use_gpu=model_spec.get("can_use_gpu", False) and env["USE_GPU"]
    )

#
# The dataset preprocessing occurs outside the experimental loops since it's deterministic.
# We also create a "Combined" dataset out of all the individual ones.
#
datasets = {}
for dataset_name, filenames in env["DATASETS"].items():
    preprocess_rule = getattr(env, "Preprocess{}".format(dataset_name))
    datasets[dataset_name] = preprocess_rule(
        "work/${DATASET_NAME}.json.gz",
        filenames,
        DATASET_NAME=dataset_name
    )

# This alias allows us to run "scons datasets"
env.Alias("datasets", list(datasets.values()))

# This function takes a dict of parameters where some may be lists (corresponding to e.g. different
# embedding sizes, learning rates, etc) and generates each unique, concrete combination of parameter
# values.  In other words, it sets up grid search.  It also creates a unique identifier string for
# each combination via a hash function.
def expand(model_spec):
    retval = [[]]
    for key, vals in model_spec.items():
        retval = sum([[old + [(key, val)] for val in (vals if isinstance(vals, list) else [vals])] for old in retval], [])    
    return {md5(str(sorted(config)).encode()).hexdigest() : dict(config) for config in retval}

#
# The main experimental (nested) loop.
#
outputs_by_config = {}
output_sets = []
for fold in range(1, int(env["FOLDS"]) + 1):
    splits = {}
    for dataset_name, dataset in datasets.items():
        splits[dataset_name] = env.SplitData(
            ["work/${{FOLD}}/${{DATASET_NAME}}/{}_split.json.gz".format(x) for x in ["train", "dev", "test"]],
            dataset,
            FOLD=fold,
            DATASET_NAME=dataset_name,
        )
    env.Alias("splits", list(splits.values()))
    for dataset_name, (train_split, dev_split, test_split) in splits.items():
        trained_models = {}
        for model_name, model_spec in env["MODELS"].items():
            training_rule = getattr(env, "Train{}".format(model_name))
            applying_rule = getattr(env, "Apply{}".format(model_name))
            for eid, conf in expand(model_spec).items():
                model, train_stats = training_rule(
                    [
                        "work/${FOLD}/${DATASET_NAME}/${MODEL_NAME}/model_${ID}.gz",
                        "work/${FOLD}/${DATASET_NAME}/${MODEL_NAME}/trainstats_${ID}.json.gz",
                    ],
                    [
                        train_split,
                        dev_split,
                    ],
                    **conf,
                    FOLD=fold,
                    DATASET_NAME=dataset_name,
                    ID=eid,
                    MODEL_NAME=model_name,
                    USE_GPU=env["USE_GPU"]
                )
                env.Alias("models", model)
                #for apply_dataset_name, (_, _, test_split) in splits.items():
                if True:
                    apply_dataset_name = dataset_name
                    output, apply_stats = applying_rule(
                        [
                            "work/${FOLD}/${DATASET_NAME}/${MODEL_NAME}/${APPLY_DATASET}/output_${ID}.json.gz",
                            "work/${FOLD}/${DATASET_NAME}/${MODEL_NAME}/${APPLY_DATASET}/applystats_${ID}.json.gz",
                        ],
                        [
                            model,
                            test_split
                        ],
                        **conf,
                        FOLD=fold,
                        DATASET_NAME=dataset_name,
                        APPLY_DATASET=apply_dataset_name,
                        ID=eid,
                        MODEL_NAME=model_name,
                        USE_GPU=env["USE_GPU"]
                    )
                    #env.Alias("outputs", output)
                    key = (dataset_name, apply_dataset_name, model_name)
                    outputs_by_config[key] = outputs_by_config.get(key, []) + [output]
                    config_file = env.SaveConfig(
                        "work/${FOLD}/${DATASET_NAME}/${MODEL_NAME}/${APPLY_DATASET}/config_${ID}.json.gz",
                        [],
                        CONFIG=json.dumps(
                            list(
                                sorted(
                                    list(conf.items()) + [
                                        ("DATASET_NAME", dataset_name),
                                        ("APPLY_DATASET", apply_dataset_name),
                                        ("MODEL_NAME", model_name),
                                        ("FOLD", fold),
                                    ]
                                )
                            )
                        ),
                        FOLD=fold,
                        DATASET_NAME=dataset_name,
                        APPLY_DATASET=apply_dataset_name,
                        ID=eid,
                        MODEL_NAME=model_name,
                    )
                    env.Alias("configs", config_file)
                    output_sets.append((config_file, output, train_stats, apply_stats))
                    #heatmaps.append(heatmap)

outputs = sum(outputs_by_config.values(), [])
env.Alias("outputs", outputs)

heatmaps = []
for (train, test, model), outputs in outputs_by_config.items():
    sheatmap, sbylength, theatmap, tbylength = env.Figures(
        [
            "work/figures/${DATASET_NAME}/${MODEL_NAME}/${APPLY_DATASET}/sentence_heatmap.png",
            "work/figures/${DATASET_NAME}/${MODEL_NAME}/${APPLY_DATASET}/sentence_bylength.png",
            "work/figures/${DATASET_NAME}/${MODEL_NAME}/${APPLY_DATASET}/token_heatmap.png",
            "work/figures/${DATASET_NAME}/${MODEL_NAME}/${APPLY_DATASET}/token_bylength.png",
        ], 
        outputs,
        DATASET_NAME=train,
        APPLY_DATASET=test,
        MODEL_NAME=model
    )
    heatmaps.append((sheatmap, sbylength, theatmap, tbylength))
    env.Alias("heatmaps", [sheatmap, theatmap])
    env.Alias("bylengths", [sbylength, tbylength])

dataset_summary = env.SummarizeDatasets("work/datasets.tex", datasets.values())

results = env.Evaluate(
    ["work/evaluation.json.gz"],
    output_sets
)

report_tex = env.CreateReport(
    ["work/report.tex"],
    [results, dataset_summary, heatmaps]
)

report = env.CompileReport(
    "work/report.pdf",
    report_tex,
)
