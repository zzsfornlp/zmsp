#

# prepare data and embeddings

# todo(warn): directly from the mdp project (old version!!)

# add the bigger ones
# 'ar bg ca zh hr cs da nl en et fi fr de he hi id it ja ko la lv no pl pt ro ru sk sl es sv uk'
#print("\n".join([z[1][0].split("-")[0].split("_")[1] for z in x]))
LANGUAGE_LIST = (
    ["ar", ["UD_Arabic-PADT"], "Afro-Asiatic.Semitic"],
    ["bg", ["UD_Bulgarian-BTB"], "IE.Slavic.South"],
    ["ca", ["UD_Catalan-AnCora"], "IE.Romance.West"],
    ["zh", ["UD_Chinese-GSD"], "Sino-Tibetan"],
    ["hr", ["UD_Croatian-SET"], "IE.Slavic.South"],
    ["cs", ["UD_Czech-PDT", "UD_Czech-CAC", "UD_Czech-CLTT", "UD_Czech-FicTree"], "IE.Slavic.West"],
    ["da", ["UD_Danish-DDT"], "IE.Germanic.North"],
    ["nl", ["UD_Dutch-Alpino", "UD_Dutch-LassySmall"], "IE.Germanic.West"],
    ["en", ["UD_English-EWT"], "IE.Germanic.West"],
    ["et", ["UD_Estonian-EDT"], "Uralic.Finnic"],
    ["fi", ["UD_Finnish-TDT"], "Uralic.Finnic"],
    ["fr", ["UD_French-GSD"], "IE.Romance.West"],
    ["de", ["UD_German-GSD"], "IE.Germanic.West"],
    ["he", ["UD_Hebrew-HTB"], "Afro-Asiatic.Semitic"],
    ["hi", ["UD_Hindi-HDTB"], "IE.Indic"],
    ["id", ["UD_Indonesian-GSD"], "Austronesian.Malayo-Sumbawan"],
    ["it", ["UD_Italian-ISDT"], "IE.Romance.Italo"],
    ["ja", ["UD_Japanese-GSD"], "Japanese"],
    ["ko", ["UD_Korean-GSD", "UD_Korean-Kaist"], "Korean"],
    ["la", ["UD_Latin-PROIEL"], "IE.Latin"],
    ["lv", ["UD_Latvian-LVTB"], "IE.Baltic"],
    ["no", ["UD_Norwegian-Bokmaal", "UD_Norwegian-Nynorsk"], "IE.Germanic.North"],
    ["pl", ["UD_Polish-LFG", "UD_Polish-SZ"], "IE.Slavic.West"],
    ["pt", ["UD_Portuguese-Bosque", "UD_Portuguese-GSD"], "IE.Romance.West"],
    ["ro", ["UD_Romanian-RRT"], "IE.Romance.East"],
    ["ru", ["UD_Russian-SynTagRus"], "IE.Slavic.East"],
    ["sk", ["UD_Slovak-SNK"], "IE.Slavic.West"],
    ["sl", ["UD_Slovenian-SSJ", "UD_Slovenian-SST"], "IE.Slavic.South"],
    ["es", ["UD_Spanish-GSD", "UD_Spanish-AnCora"], "IE.Romance.West"],
    ["sv", ["UD_Swedish-Talbanken"], "IE.Germanic.North"],
    ["uk", ["UD_Ukrainian-IU"], "IE.Slavic.East"],
)

TRAIN_LANG = "en"

# confs
UD2_DIR = "../data/ud-treebanks-v2.2/"
OUT_DIR = "./data2.2_more/"
LIB_DIR = "./data2.2_more/fastText_multilingual/"

# ===== help
import os, subprocess, sys, gzip

sys.path.append(LIB_DIR)        # project embeddings

from fasttext import FastVector

printing = lambda x: print(x, file=sys.stderr, flush=True)

def system(cmd, pp=False, ass=False, popen=False):
    if pp:
        printing("Executing cmd: %s" % cmd)
    if popen:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        n = p.wait()
        output = str(p.stdout.read().decode())
    else:
        n = os.system(cmd)
        output = None
    if pp:
        printing("Output is: %s" % output)
    if ass:
        assert n==0
    return output

def zopen(filename, mode='r', encoding="utf-8"):
    if filename.endswith('.gz'):
        # "t" for text mode of gzip
        return gzip.open(filename, mode+"t", encoding=encoding)
    else:
        return open(filename, mode, encoding=encoding)
# =====

#
def deal_conll_file(fin, fout):
    for line in fin:
        line = line.strip()
        fields = line.split("\t")
        if len(line) == 0:
            fout.write("\n")
        else:
            try:
                z = int(fields[0])
                fields[4] = fields[3]
                fields[3] = "_"
                fout.write("\t".join(fields)+"\n")
            except:
                pass

#
def main():
    # first get the English one
    lang = "en"
    system("wget -nc -O %s/wiki.%s.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.%s.vec" % (OUT_DIR, lang, lang), pp=True)
    # en_dict = FastVector(vector_file='%s/wiki.en.vec' % OUT_DIR)
    for zzz in LANGUAGE_LIST:
        lang, fnames = zzz[0], zzz[1]
        printing("Dealing with lang %s." % lang)
        for curf in ["train", "dev", "test"]:
            out_fname = "%s/%s_%s.conllu" % (OUT_DIR, lang, curf)
            fout = zopen(out_fname, "w")
            for fname in fnames:
                last_name = fname.split("-")[-1].lower()
                path_name = "%s/%s/%s_%s-ud-%s.conllu" % (UD2_DIR, fname, lang, last_name, curf)
                if os.path.exists(path_name):
                    with zopen(path_name) as fin:
                        deal_conll_file(fin, fout)
            fout.close()
            # stat
            system('cat %s | grep -E "^$" | wc' % out_fname, pp=True)
            system('cat %s | grep -Ev "^$" | wc' % out_fname, pp=True)
            system("cat %s | grep -Ev '^$' | cut -f 5 -d $'\t'| grep -Ev 'PUNCT|SYM' | wc" % out_fname, pp=True)
        # get original embed
        system("wget -nc -O %s/wiki.%s.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.%s.vec" % (OUT_DIR, lang, lang), pp=True)
        # project with LIB-matrix
        lang_dict = FastVector(vector_file='%s/wiki.%s.vec' % (OUT_DIR, lang))
        lang_dict.apply_transform("%s/alignment_matrices/%s.txt" % (LIB_DIR, lang))
        lang_dict.export("%s/wiki.multi.%s.vec" % (OUT_DIR, lang))

if __name__ == '__main__':
    main()

# python3 zprep_ud22.py |& grep -v "s$" | tee data2.2_more/log
