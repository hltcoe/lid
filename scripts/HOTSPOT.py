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
import math

warnings.simplefilter(action='ignore')


def train_model(train, dev, output, rest):
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotspot_path", dest="hotspot_path")
    args, rest = parser.parse_known_args(rest)
    tmp = tempfile.mkdtemp()

    try:
        instances = {}
        for item in train:
            text = " ".join([t["form"] for t in item["tokens"]])
            lang = item["tokens"][0]["language"]
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

        command = "java -cp {} hotspot/trainer/BuildConfigFile -a {} -c {}/config -d {} -t 1 -p all".format(args.hotspot_path, temp, config, data)
        print(command)
        pid = Popen(shlex.split(command), stderr=PIPE, stdout=PIPE)
        pid.communicate()
        model_out = os.path.join(tmp, "model.ascii")
        command = "java -Xmx12g -cp {} hotspot/trainer/HotspotTrainer -c {}/config_med -a {}".format(args.hotspot_path, config, model_out)
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
    logging.info("Using temporary path: %s", tmp)
    expand = 30
    try:
        data = [" ".join([t["form"] for t in x["tokens"]]) for x in test]
        print(len(data))
        asc = os.path.join(tmp, "config.ascii")
        with open(asc, "wb") as ofd:
            ofd.write(model)
        binary = os.path.join(tmp, "config.bin")
        command = "java -Xmx4g -cp {} hotspot/scanner/HotspotScanner -D  -a {} -c -K -N -s {} -E {}".format(args.hotspot_path, asc, binary, expand)
        pid = Popen(shlex.split(command), stderr=PIPE, stdout=PIPE)
        out, err = pid.communicate()
        datafile = os.path.join(tmp, "data.txt")
        with open(datafile, "wt") as ofd:
            ofd.write("\n".join(data) + "\n")
        command = "java -Xmx4g -cp {} hotspot/scanner/HotspotScanner -D -K -c -b {} -i {}".format(args.hotspot_path, binary, datafile)
        pid = Popen(shlex.split(command), stderr=PIPE, stdout=PIPE)
        out, err = pid.communicate()
        for m in re.finditer(r"^(\d+)\:[^\n]*\n([^\n]*)\n\s*\n", out.decode(), flags=re.M|re.S):
            i = int(m.group(1))            
            line = m.group(2)
            scores = [x.split(":") for x in line.strip().rstrip("|").split("&") if not re.match(r"^\s*$", x)]
            test[i - 1]["scores"] = {k : math.log(float(v) + 0.0000001) for k, v in scores}


        # word level
        data = sum([[t["form"] for t in x["tokens"]] for x in test], [])
        asc = os.path.join(tmp, "config.ascii")
        binary = os.path.join(tmp, "config.bin")
        command = "java -Xmx4g -cp {} hotspot/scanner/HotspotScanner -D -a {} -c -K -x -N -s {} -E {}".format(args.hotspot_path, asc, binary, expand)
        pid = Popen(shlex.split(command), stderr=PIPE, stdout=PIPE)
        out, err = pid.communicate()
        datafile = os.path.join(tmp, "data.txt")
        with open(datafile, "wt") as ofd:
            ofd.write("\n".join([t for t in data]) + "\n")
        command = "java -Xmx4g -cp {} hotspot/scanner/HotspotScanner -D -K -c -b {} -i {}".format(args.hotspot_path, binary, datafile)
        pid = Popen(shlex.split(command), stderr=PIPE, stdout=PIPE)
        out, err = pid.communicate()
        mapping = {int(m.group(1)) : m.group(2) for m in re.finditer(r"^(\d+)\:[^\n]*\n([^\n]*)\n\s*\n", out.decode(), flags=re.M|re.S)}
        k = 1
        for i in range(len(test)):
            for j in range(len(test[i]["tokens"])):                
                line = mapping.get(k, "unknown:0&|")
                scores = [x.split(":") for x in line.strip().rstrip("|").split("&") if not re.match(r"^\s*$", x)]
                test[i]["tokens"][j]["scores"] = {k : math.log(float(v) + 0.0000000001) for k, v in scores}
                k += 1
    finally:
        shutil.rmtree(tmp)
    return test
