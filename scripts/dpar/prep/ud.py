#!/usr/bin/env python3

# prepare dependencies for UD 2.1
# simply analysing and ranking the Treebanks
import os, subprocess

UD_HOME = os.path.abspath("../UD/")
OUT_DIR = "./"
CONVERT_PYFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conll.py")

FILES = ["train", "dev", "test"]
printing = lambda x: print(x, flush=True)

def system(cmd, pp=True, ass=True, popen=False):
    if pp:
        printing("Executing cmd: %s" % cmd)
    if popen:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        n = p.wait()
        output = p.stdout.read()
    else:
        n = os.system(cmd)
        output = None
    if pp:
        printing("Output is: %s" % output)
    if ass:
        assert n==0
    return output

def read_analysis(output):
    output = output.decode("utf-8")
    fs = output.split("\n")[-2].split()
    assert fs[0] == "results:"
    return {"sents":int(fs[1]), "sprate":float(fs[3]), "words":int(fs[4]), "wprate":float(fs[6])}

def none_analysis():
    return {"sents":0, "sprate":0., "words":0, "wprate":0.}

def main():
    # info
    dirs = os.listdir(UD_HOME)
    dirs = sorted(dirs)
    # -- collect info
    infos = []
    for name in dirs:
        info = {}
        cur_dir = UD_HOME + "/" + name + "/"
        info["DIR"] = cur_dir
        info["NAME"] = name
        info["LANG"] = None
        files = os.listdir(cur_dir)
        for f in files:
            fs = f.split("-")
            if len(fs) == 3 and fs[1] == "ud":
                for which in FILES:
                    if fs[-1] == which+".conllu":
                        if info["LANG"] is None:
                            info["LANG"] = fs[0]
                        assert info["LANG"] == fs[0]
                        info[which] = cur_dir + f
        infos.append(info)
    # -- analyse and restore info
    count_all = 0
    count_hastrain = 0
    for info in infos:
        count_all += 1
        to_dir = OUT_DIR + info["NAME"]
        system("mkdir -p %s" % to_dir)
        for which in FILES:
            if which in info:
                # link data
                system("ln -s %s %s/%s.conll" % (info[which], to_dir, which), ass=False)
        # analyse for train
        if "train" in info:
            count_hastrain += 1
            output = system("python3 %s %s conll06" % (CONVERT_PYFILE, info["train"]), popen=True)
            info["rr"] = read_analysis(output)
        else:
            info["rr"] = none_analysis()
    # -- rank them
    printing("ALL/HASTRAIN=%s/%s" % (count_all, count_hastrain))
    for k in none_analysis():
        printing("\nRanking on %s" % k)
        for i, one in enumerate(sorted(infos, key=lambda x: x["rr"][k], reverse=True)):
            printing("#%s-%s: %s" % (i+1, k, one))

if __name__ == '__main__':
    main()

# for d in *;
# do
#     echo $d;
#     cd $d;
#     for name in train dev test;
#     do ln -s $name.conll $name.auto;
#     done
#     cd ..;
# done
