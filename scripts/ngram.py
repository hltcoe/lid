import argparse
import tempfile
import re
import gzip
import os
import os.path
import shutil
from subprocess import Popen, PIPE
import shlex
import logging
from glob import glob
import warnings
import sys
import xml.etree.ElementTree as et

warnings.simplefilter(action='ignore')



def train_model(train, dev, output, rest):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngram_path", dest="ngram_path")
    parser.add_argument("--ngram_length", dest="ngram_length", type=int)
    args, rest = parser.parse_known_args(rest)
    tmp = tempfile.mkdtemp()

    try:
        with open(os.path.join(tmp, "train.txt"), "wt") as ofd:
            for item in train:
                counts = {}
                for tok in item["tokens"]:
                    counts[tok["language"]] = counts.get(tok["language"], 0) + 1
                maj_lang = sorted([(v, k) for k, v in counts.items()])[-1][1]
                text = " ".join([t["form"] for t in item["tokens"]])
                ofd.write("{}\t{}\t{}\n".format(item["id"], maj_lang, text))
        command = "stack exec ngramClassifier -- train --trainFile {} --n {} --modelFile {}".format(os.path.join(tmp, "train.txt"), args.ngram_length, os.path.join(tmp, "model.bin"))
        #command = "java -cp {}/Hotspot_Parent/Hotspot_Parent/hotspot/target/hotspot-8.17-SNAPSHOT-jar-with-dependencies.jar hotspot/trainer/BuildConfigFile -a {} -c {}/config -d {} -t 1 -p all".format(args.hotspot_path, temp, config, data)
        print(command)
        pid = Popen(shlex.split(command), stderr=PIPE, stdout=PIPE, cwd=args.ngram_path)
        out, err = pid.communicate()
        #model_out = os.path.join(tmp, "model.ascii")
        #command = "java -Xmx12g -cp {}/Hotspot_Parent/Hotspot_Parent/hotspot/target/hotspot-8.17-SNAPSHOT-jar-with-dependencies.jar hotspot/trainer/HotspotTrainer -c {}/config_med -a {}".format(args.hotspot_path, config, model_out)
        #print(command)
        #pid = Popen(shlex.split(command), stderr=PIPE, stdout=PIPE)
        #out, err = pid.communicate()
        if pid.returncode != 0:
            raise Exception(err + out)
        with open(os.path.join(tmp, "model.bin"), "rb") as ifd:
            model = ifd.read()
        return model
    finally:
        shutil.rmtree(tmp)


def apply_model(model, test, args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngram_path", dest="ngram_path")
    parser.add_argument("--ngram_length", dest="ngram_length", type=int)
    args, rest = parser.parse_known_args(args)
    tmp = tempfile.mkdtemp()
    print(tmp)
    expand = 30
    try:
        with open(os.path.join(tmp, "model.bin"), "wb") as ofd:
            ofd.write(model)
        with open(os.path.join(tmp, "stest.txt"), "wt") as s_ofd, open(os.path.join(tmp, "wtest.txt"), "wt") as w_ofd:
            for item in test:
                counts = {}
                for tid, tok in enumerate(item["tokens"]):
                    counts[tok["language"]] = counts.get(tok["language"], 0) + 1
                    w_ofd.write("{}{}\t{}\t{}\n".format(item["id"], tid, tok["language"], tok["form"]))
                maj_lang = sorted([(v, k) for k, v in counts.items()])[-1][1]
                text = " ".join([t["form"] for t in item["tokens"]])
                s_ofd.write("{}\t{}\t{}\n".format(item["id"], maj_lang, text))
        command = "stack exec ngramClassifier -- apply --testFile {} --n {} --modelFile {} --scoresFile {}".format(os.path.join(tmp, "stest.txt"), args.ngram_length, os.path.join(tmp, "model.bin"), os.path.join(tmp, "sscores.txt"))
        print(command)
        pid = Popen(shlex.split(command), stderr=PIPE, stdout=PIPE, cwd=args.ngram_path)
        out, err = pid.communicate()
        if pid.returncode != 0:
            raise Exception(err + out)        
        command = "stack exec ngramClassifier -- apply --testFile {} --n {} --modelFile {} --scoresFile {}".format(os.path.join(tmp, "wtest.txt"), args.ngram_length, os.path.join(tmp, "model.bin"), os.path.join(tmp, "wscores.txt"))
        print(command)
        pid = Popen(shlex.split(command), stderr=PIPE, stdout=PIPE, cwd=args.ngram_path)
        out, err = pid.communicate()
        if pid.returncode != 0:
            raise Exception(err + out)        
        with open(os.path.join(tmp, "sscores.txt"), "rt") as s_ifd, open(os.path.join(tmp, "wscores.txt"), "rt") as w_ifd:
            for i, line in enumerate(s_ifd):
                _, _, _, scores = line.strip().split("\t")
                scores = {k : float(v) for k, v in [x.split("=") for x in scores.split(" ")]}
                test[i]["scores"] = scores
                for j in range(len(test[i]["tokens"])):
                    wline = w_ifd.readline()
                    _, _, _, scores = wline.strip().split("\t")
                    scores = {k : float(v) for k, v in [x.split("=") for x in scores.split(" ")]}
                    test[i]["tokens"][j]["scores"] = scores
    finally:
        shutil.rmtree(tmp)
    return test
    #sys.exit()
    #return (cids, golds, guesses)


    # tmp = tempfile.mkdtemp()
    # try:
    #     instances = []
    #     cids = []
    #     #with gzip.open(args.input, "rt") as ifd:
    #     #    for line in ifd:
    #     #        cid, label, text = line.strip().split("\t")
    #     for item in test:
    #         instances.append((item["label"], item["sequence"]))
    #         cids.append(item["id"])

    #     dev_file = os.path.join(tmp, "dev.txt")
    #     with open(dev_file, "wt") as ofd:
    #         ofd.write("\n".join([t for _, t in instances]))

    #     command = "java -Xmx12g -cp {} hotspot/scanner/HotspotScanner -b {} -i {}".format(args.hotspot_path, args.model, dev_file)
    #     logging.info("Running '%s'", command)
    #     pid = Popen(shlex.split(command), stdout=PIPE, stderr=PIPE)
    #     out, err = pid.communicate()
    #     if pid.returncode != 0:
    #         raise Exception(err)
    #     guesses = []
    #     #for line in [l for l in out.decode().split("\n") if l.startswith("\t")]:
    #     #    guesses.append(line.strip().split(":")[0])
    #     #gold = [l for l, _ in instances]
    #     #with gzip.open(args.output, "wt") as ofd:
    #     #    for c, l, g in zip(cids, gold, guesses):
    #     #        ofd.write("{}\t{}\t{}\n".format(c, l, g))

    #     #return f1_score(y_true=gold, y_pred=guesses, average="macro")
    # finally:
    #     shutil.rmtree(tmp)        

        
    # print(rest)
    # sys.exit()
    # best_acc = 0.0
    # best_size = None
    # best_exp = None
    # for size in ["fast", "med", "lrg"]:
    #     #logging.info("Building %s model", size)
    #     train_one_model(args, size, tmp)
    #     for exp in [16, 32, 64, 128, 256]:
    #         #logging.info("...with %d expansion", exp)
    #         command = "java -Xmx12g -cp {} hotspot/scanner/HotspotScanner -a {} -s {} -E {}".format(args.jar,
    #                                                                                                 os.path.join(tmp, "model.ascii"),
    #                                                                                                 os.path.join(tmp, "model.bin"),
    #                                                                                                 exp)
    #         logging.info("Running '%s'", command)            
    #         pid = Popen(shlex.split(command), stderr=PIPE, stdout=PIPE)
    #         out, err = pid.communicate()
    #         if pid.returncode != 0:
    #             raise Exception(err)
    #         args.input = args.dev
    #         args.model = os.path.join(tmp, "model.bin")
    #         acc = apply_model(args, tmp)
    #         #logging.info("%s", acc)
    #         if acc > best_acc:
    #             best_acc = acc
    #             best_size = size
    #             best_exp = exp
                
    #             shutil.copyfile(args.model, args.model + ".best")
    #             #sys.exit()
    #             #shutil.copyfile(os.path.join(tmp, "model.ascii"), args.output + ".ascii")
    # shutil.copyfile(args.model + ".best", args.output)                
    # print(best_acc, best_size, best_exp)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("-t", "--train", dest="train", help="Input file")
    train_parser.add_argument("-d", "--dev", dest="dev", help="Input file")
    train_parser.add_argument("-o", "--output", dest="output", help="Output file")
    train_parser.add_argument("-j", "--jar", dest="jar", default="/home/tom/hotspot/Hotspot_Parent/Hotspot_Parent/hotspot/target/hotspot-scanner-jar-with-dependencies.jar", help="Jar file")
    train_parser.set_defaults(func=train_model)
    
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("-m", "--model", dest="model", help="Model file")
    apply_parser.add_argument("-i", "--input", dest="input", help="Input file")
    apply_parser.add_argument("-o", "--output", dest="output", help="Output file")
    apply_parser.add_argument("-j", "--jar", dest="jar", default="/home/tom/hotspot/Hotspot_Parent/Hotspot_Parent/hotspot/target/hotspot-scanner-jar-with-dependencies.jar", help="Jar file")    
    apply_parser.set_defaults(func=apply_model)
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    tmp = tempfile.mkdtemp()
    args.func(args, tmp)
    shutil.rmtree(tmp)
