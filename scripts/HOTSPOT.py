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
#from sklearn.metrics import f1_score
import warnings
import sys

warnings.simplefilter(action='ignore')


def train_model(train, dev, output, rest):
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotspot_path", dest="hotspot_path")
    args, rest = parser.parse_known_args(rest)
    tmp = tempfile.mkdtemp()

    try:
        instances = {}
        for item in train:
            lang = item["label"]
            text = item["sequence"]
            instances[lang] = instances.get(lang, [])
            instances[lang].append(text)

        data, config, temp = [os.path.join(tmp, x) for x in ["data", "config", "temp"]]
        [os.mkdir(x) for x in [data, config, temp]]
        for label, texts in instances.items():
            os.mkdir(os.path.join(data, label))
            for i, text in enumerate(texts):
                with open(os.path.join(data, label, "{}-{}-UTF-8".format(i, label)), "wt") as ofd:
                    try:
                        ofd.write(text)
                    except:
                        ofd.write(text.encode())

        command = "java -cp {}/Hotspot_Parent/Hotspot_Parent/hotspot/target/hotspot-8.17-SNAPSHOT-jar-with-dependencies.jar hotspot/trainer/BuildConfigFile -a {} -c {}/config -d {} -t 1 -p all".format(args.hotspot_path, temp, config, data)
        print(command)
        pid = Popen(shlex.split(command), stderr=PIPE, stdout=PIPE)
        pid.communicate()
        model_out = os.path.join(tmp, "model.ascii")
        command = "java -Xmx12g -cp {}/Hotspot_Parent/Hotspot_Parent/hotspot/target/hotspot-8.17-SNAPSHOT-jar-with-dependencies.jar hotspot/trainer/HotspotTrainer -c {}/config_med -a {}".format(args.hotspot_path, config, model_out)
        print(command)
        pid = Popen(shlex.split(command), stderr=PIPE, stdout=PIPE)
        out, err = pid.communicate()
        if pid.returncode != 0:
            raise Exception(err + out)
        with open(model_out, "rb") as ifd:
            model = ifd.read()
        return model
    finally:
        shutil.rmtree(tmp)


def apply_model(model, test, args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotspot_path", dest="hotspot_path")
    args, rest = parser.parse_known_args(args)
    tmp = tempfile.mkdtemp()
    print(tmp)
    expand = 30
    try:
        data = [x["sequence"] for x in test]
        asc = os.path.join(tmp, "config.ascii")
        with open(asc, "wb") as ofd:
            ofd.write(model)
        binary = os.path.join(tmp, "config.bin")
        command = "java -Xmx4g -cp {}/Hotspot_Parent/Hotspot_Parent/hotspot/target/hotspot-8.17-SNAPSHOT-jar-with-dependencies.jar hotspot/scanner/HotspotScanner -D -a {} -s {} -E {}".format(args.hotspot_path, asc, binary, expand)
        print(command)
        pid = Popen(shlex.split(command), stderr=PIPE, stdout=PIPE)
        out, err = pid.communicate()
        datafile = os.path.join(tmp, "data.txt")
        with open(datafile, "wt") as ofd:
            for t in data:
                try:
                    ofd.write(t + "\n")
                except:
                    ofd.write(t.encode() + "\n")
            #ofd.write("\n".join([t for t in data]) + "\n")
        command = "java -Xmx4g -cp {}/Hotspot_Parent/Hotspot_Parent/hotspot/target/hotspot-8.17-SNAPSHOT-jar-with-dependencies.jar hotspot/scanner/HotspotScanner -D -b {} -i {}".format(args.hotspot_path, binary, datafile)
        print(command)
        #sys.exit()
        pid = Popen(shlex.split(command), stderr=PIPE, stdout=PIPE)
        out, err = pid.communicate()
        #print(out.decode())
        #sys.exit()
        #print(out)
        #lines = out.decode("utf-8") #.split("\n")[2:]
        #print(len(test))
        #print(len(data))
        #print(err.decode("utf-8"))
        #sys.exit()
        #cids, guesses, golds = [], [], []
        #with open("a.out", "wt") as ofd:
        #    ofd.write("\n".join([t for t in data]) + "\n")
        #with open("b.out", "wt") as ofd:
        #    ofd.write("\n".join(lines))
        for i in range(len(test)):
            test[i]["scores"] = {}

        for m in re.finditer(r"^(\d+)\:[^\n]*\n([^\n]*)\n\s*\n", out.decode(), flags=re.M|re.S):
        #for i, item in enumerate(test): #(cid, lang, _) in enumerate(data):
        #    print(i)
            #print(i, item, lines[])
            i = int(m.group(1))            
            line = m.group(2)
            #print(i, m.group(2))
            #line = lines[1 + (i * 3)] #.strip().split(":")[0]
            #print(line)
            scores = [x.split(":") for x in line.strip().rstrip("|").split("&") if not re.match(r"^\s*$", x)]
            #print(scores)
            test[i - 1]["scores"] = {k : float(v) for k, v in scores}
            #guesses.append(guess)
            #golds.append(lang)
            #cids.append(cid)
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
