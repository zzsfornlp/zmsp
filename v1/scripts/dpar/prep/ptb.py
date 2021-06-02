#!/usr/bin/env python3

# prepare dependencies for CTB with Stanford Dependencies
# -> https://nlp.stanford.edu/software/dependencies_manual.pdf
# -> stanford parser 3.3.0 + CoreNLP 3.8.0
import os, sys, subprocess

# assuming in specific-data-dir
BASE_DIR="."
BR_HOME = "../PTB/parsed/mrg/wsj/"
OUT_DIR = "./"
dir_name = os.path.dirname(os.path.abspath(__file__))
CONVERT_PYFILE = os.path.join(dir_name, "conll.py")

DEP_CONVERTER = "../../tools/stanford-parser.jar"
PENN2MALT_JAR = "../../tools/Penn2Malt.jar"
ENG_P2M_HEADRULES = "../../tools/headrules.txt"

TAGGER = "../../tools/stanford-corenlp-3.8.0.jar"
TAGGER_PROP = """
                    arch = bidirectional5words,naacl2003unknowns
            wordFunction = edu.stanford.nlp.process.AmericanizeFunction
         closedClassTags =
 closedClassTagThreshold = 40
 curWordMinFeatureThresh = 2
                   debug = false
             debugPrefix =
            tagSeparator = newline
                encoding = UTF-8
              iterations = 100
                    lang = english
    learnClosedClassTags = false
        minFeatureThresh = 2
           openClassTags =
rareWordMinFeatureThresh = 5
          rareWordThresh = 5
                  search = qn
                    sgml = false
            sigmaSquared = 0.5
                   regL1 = 0.75
               tagInside =
                tokenize = false
        tokenizerFactory =
        tokenizerOptions =
                 verbose = false
          verboseResults = false
    veryCommonWordThresh = 250
                xmlInput =
              outputFile =
            outputFormat = tsv
     outputFormatOptions =
                nthreads = 4
"""

def printing(s):
    print(s, flush=True)

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

# def system(cmd, pp=False, ass=True):
#     if pp:
#         printing("Executing cmd: %s" % cmd)
#     n = os.system(cmd)
#     if ass:
#         assert n==0

def convert_sd(in_name, out_name):
    system("java -cp %s -mx1g edu.stanford.nlp.trees.EnglishGrammaticalStructure -treeFile %s -conllx -basic > %s"
           % (DEP_CONVERTER, in_name, out_name))

def convert_p2m(in_name, out_name):
    # p2m generates: *.3.pa.tab/pos/dep
    system("java -jar %s %s %s 3 2 penn" % (PENN2MALT_JAR, in_name,ENG_P2M_HEADRULES))
    # tab2conll
    system("python3 %s %s tab %s conll06" % (CONVERT_PYFILE, in_name+".3.pa.gs.tab", out_name))

# -------------

def get_name(s, N=2):
    n_zero = N - len(s)
    assert n_zero >= 0
    for i in range(n_zero):
        s = "0" + s
    return s

def get_pairs(nway, train, dev, test):
    pairs = []
    assert len(train) % nway == 0
    for i in range(nway):
        tr, tt = [], []
        for one in train:
            if (one-train[0]) // (len(train)//nway) == i:
                tt.append(one)
            else:
                tr.append(one)
        pairs.append((tr, tt))
    pairs.append((train, dev+test))
    return pairs

"""
*.ptb: original bracketed files
*.conll: conll06 dp file (gold pos)
*.pos: form/pos file (gold pos)
*.tag: form/pos file (auto pos)
*.gold: gold tr/dev/test files
*.auto: auto tr/dev/test files
"""
def main(from_step, dp_convert):
    all_sections = list(range(25))
    train = [i for i in range(2, 21+1)]
    dev = [22]
    test = [23]
    printing("From step %s" % from_step)
    # step0: from PTB to *.conll *.pos
    if from_step<=0:
        printing("Step0: convert 0-24 files")
        for i in all_sections:
            nn = get_name(str(i))
            FILE_BASE = OUT_DIR+nn
            system("cat %s/%s/*.mrg >%s" % (BR_HOME, nn, FILE_BASE+".ptb"))
            dp_convert(FILE_BASE+".ptb", FILE_BASE+".conll")
            system("python3 %s %s conll06 %s pos" % (CONVERT_PYFILE, FILE_BASE+".conll", FILE_BASE+".pos"))
    # step1: tagging, *.tag
    if from_step<=1:
        printing("Step1: tagging them")
        paris = get_pairs(10, train, dev, test)
        PROP = OUT_DIR+"props"
        with open(PROP, "w") as fd:
            fd.write(TAGGER_PROP)
        for tr, tt in paris:
            printing("Training on %s, tagging on %s." % (tr, tt))
            trn, ttn = [get_name(str(i)) for i in tr], [get_name(str(i)) for i in tt]
            # concat training files
            NAME_BASE = OUT_DIR+"TR%s-TT%s_%s" % (len(trn), len(ttn), "".join(ttn))
            TRAIN_NAME = NAME_BASE+".pos"
            MODEL_NAME = NAME_BASE+".model"
            system("cat %s > %s" % (" ".join([OUT_DIR+z+".pos" for z in trn]), TRAIN_NAME))
            # train
            system('java -cp %s edu.stanford.nlp.tagger.maxent.MaxentTagger -prop %s -trainFile "format=TSV,wordColumn=0,tagColumn=1,%s" -model %s 2>%s.log' % (TAGGER, PROP, TRAIN_NAME, MODEL_NAME, MODEL_NAME))
            # test / tag
            for one in ttn:
                TEST_NAME = OUT_DIR+one+".pos"
                OUT_NAME = OUT_DIR+one+".tag"
                system('java -cp %s edu.stanford.nlp.tagger.maxent.MaxentTagger -testFile "format=TSV,wordColumn=0,tagColumn=1,%s" -model %s' % (TAGGER, TEST_NAME, MODEL_NAME))
                system('java -cp %s edu.stanford.nlp.tagger.maxent.MaxentTagger -textFile "format=TSV,wordColumn=0,tagColumn=1,%s" -model %s -tokenize false -outputFormat tsv > %s' % (TAGGER, TEST_NAME, MODEL_NAME, OUT_NAME))
    # step2: final settings, *.gold *.tag *.auto
    if from_step <= 2:
        printing("Step2: final step")
        for name, ll in (["train", train], ["dev", dev], ["test", test]):
            system("cat %s > %s" % (" ".join([OUT_DIR+get_name(str(z))+".conll" for z in ll]), name+".gold"))
            system("cat %s > %s" % (" ".join([OUT_DIR+get_name(str(z))+".tag" for z in ll]), name+".tag"))
            system("python3 %s %s.gold conll06 %s.auto conll06 %s.tag pos" % (CONVERT_PYFILE, name, name, name))

if __name__ == '__main__':
    #
    converter = "sd"
    try:
        if sys.argv[1] in ["sd", "p2m"]:
            converter = sys.argv[1]
    except:
        pass
    dp_convert = {"sd": convert_sd, "p2m": convert_p2m}[converter]
    printing("Using converter of %s: %s" % (converter, dp_convert))
    #
    from_step = 0
    try:
        x = int(sys.argv[1])
        from_step = x
    except:
        pass
    main(from_step, dp_convert)
