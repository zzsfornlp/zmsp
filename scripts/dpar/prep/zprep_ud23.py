#

# updated: prepare for UDv2.3, together with conllu_reader
# only one treebank for each language

# todo(warn): the first train-treebank is the main one
LANGUAGE_LIST = (
    ['af', 'Afrikaans', "IE.Germanic", ['UD_Afrikaans-AfriBooms'], ['UD_Afrikaans-AfriBooms'], ['UD_Afrikaans-AfriBooms']],
    ['ar', 'Arabic', "Afro-Asiatic.Semitic", ['UD_Arabic-PADT'], ['UD_Arabic-PADT'], ['UD_Arabic-PADT', 'UD_Arabic-PUD']],
    ['be', 'Belarusian', "IE.Slavic", ['UD_Belarusian-HSE'], ['UD_Belarusian-HSE'], ['UD_Belarusian-HSE']],
    ['bg', 'Bulgarian', "IE.Slavic", ['UD_Bulgarian-BTB'], ['UD_Bulgarian-BTB'], ['UD_Bulgarian-BTB']],
    ['ca', 'Catalan', "IE.Romance", ['UD_Catalan-AnCora'], ['UD_Catalan-AnCora'], ['UD_Catalan-AnCora']],
    ['cs', 'Czech', "IE.Slavic", ['UD_Czech-PDT', 'UD_Czech-CAC', 'UD_Czech-CLTT', 'UD_Czech-FicTree'], ['UD_Czech-PDT', 'UD_Czech-CAC', 'UD_Czech-CLTT', 'UD_Czech-FicTree'], ['UD_Czech-PDT', 'UD_Czech-CAC', 'UD_Czech-CLTT', 'UD_Czech-FicTree', 'UD_Czech-PUD']],
    ['da', 'Danish', "IE.Germanic", ['UD_Danish-DDT'], ['UD_Danish-DDT'], ['UD_Danish-DDT']],
    ['de', 'German', "IE.Germanic", ['UD_German-GSD'], ['UD_German-GSD'], ['UD_German-GSD', 'UD_German-PUD']],
    ['el', 'Greek', "IE.Greek", ['UD_Greek-GDT'], ['UD_Greek-GDT'], ['UD_Greek-GDT']],
    ['en', 'English', "IE.Germanic", ['UD_English-EWT', 'UD_English-GUM', 'UD_English-LinES', 'UD_English-ParTUT'], ['UD_English-EWT', 'UD_English-GUM', 'UD_English-LinES', 'UD_English-ParTUT'], ['UD_English-EWT', 'UD_English-GUM', 'UD_English-LinES', 'UD_English-ParTUT', 'UD_English-PUD']],
    ['es', 'Spanish', "IE.Romance", ['UD_Spanish-AnCora', 'UD_Spanish-GSD'], ['UD_Spanish-AnCora', 'UD_Spanish-GSD'], ['UD_Spanish-AnCora', 'UD_Spanish-GSD', 'UD_Spanish-PUD']],
    ['et', 'Estonian', "Uralic.Finnic", ['UD_Estonian-EDT'], ['UD_Estonian-EDT'], ['UD_Estonian-EDT']],
    ['eu', 'Basque', "Basque", ['UD_Basque-BDT'], ['UD_Basque-BDT'], ['UD_Basque-BDT']],
    ['fa', 'Persian', "IE.Iranian", ['UD_Persian-Seraji'], ['UD_Persian-Seraji'], ['UD_Persian-Seraji']],
    ['fi', 'Finnish', "Uralic.Finnic", ['UD_Finnish-TDT', 'UD_Finnish-FTB'], ['UD_Finnish-TDT', 'UD_Finnish-FTB'], ['UD_Finnish-TDT', 'UD_Finnish-FTB', 'UD_Finnish-PUD']],
    ['fr', 'French', "IE.Romance", ['UD_French-GSD', 'UD_French-ParTUT', 'UD_French-Sequoia', 'UD_French-Spoken'], ['UD_French-GSD', 'UD_French-ParTUT', 'UD_French-Sequoia', 'UD_French-Spoken'], ['UD_French-GSD', 'UD_French-ParTUT', 'UD_French-PUD', 'UD_French-Sequoia', 'UD_French-Spoken']],
    ['gl', 'Galician', "IE.Romance", ['UD_Galician-CTG', 'UD_Galician-TreeGal'], ['UD_Galician-CTG'], ['UD_Galician-CTG', 'UD_Galician-TreeGal']],
    ['he', 'Hebrew', "Afro-Asiatic.Semitic", ['UD_Hebrew-HTB'], ['UD_Hebrew-HTB'], ['UD_Hebrew-HTB']],
    ['hi', 'Hindi', "IE.Indic", ['UD_Hindi-HDTB'], ['UD_Hindi-HDTB'], ['UD_Hindi-HDTB', 'UD_Hindi-PUD']],
    ['hr', 'Croatian', "IE.Slavic", ['UD_Croatian-SET'], ['UD_Croatian-SET'], ['UD_Croatian-SET']],
    ['hu', 'Hungarian', "Uralic.Ugric", ['UD_Hungarian-Szeged'], ['UD_Hungarian-Szeged'], ['UD_Hungarian-Szeged']],
    # ['hy', 'Armenian', ['UD_Armenian-ArmTDP'], [], ['UD_Armenian-ArmTDP']],
    ['hy', 'Armenian', "IE.Armenian", ['UD_Armenian-ArmTDP'], ['UD_Armenian-ArmTDP'], ['UD_Armenian-ArmTDP']],
    ['id', 'Indonesian', "Austronesian.Malayo-Sumbawan", ['UD_Indonesian-GSD'], ['UD_Indonesian-GSD'], ['UD_Indonesian-GSD', 'UD_Indonesian-PUD']],
    ['it', 'Italian', "IE.Romance", ['UD_Italian-ISDT', 'UD_Italian-ParTUT', 'UD_Italian-PoSTWITA'], ['UD_Italian-ISDT', 'UD_Italian-ParTUT', 'UD_Italian-PoSTWITA'], ['UD_Italian-ISDT', 'UD_Italian-ParTUT', 'UD_Italian-PoSTWITA', 'UD_Italian-PUD']],
    ['ja', 'Japanese', "Japanese", ['UD_Japanese-GSD'], ['UD_Japanese-GSD'], ['UD_Japanese-GSD', 'UD_Japanese-Modern', 'UD_Japanese-PUD']],
    # ['kk', 'Kazakh', ['UD_Kazakh-KTB'], [], ['UD_Kazakh-KTB']],
    ['kk', 'Kazakh', " Turkic.Northwestern", ['UD_Kazakh-KTB'], ['UD_Kazakh-KTB'], ['UD_Kazakh-KTB']],
    ['ko', 'Korean', "Korean", ['UD_Korean-GSD', 'UD_Korean-Kaist'], ['UD_Korean-GSD', 'UD_Korean-Kaist'], ['UD_Korean-GSD', 'UD_Korean-Kaist', 'UD_Korean-PUD']],
    ['la', 'Latin', "IE.Latin", ['UD_Latin-ITTB', 'UD_Latin-Perseus', 'UD_Latin-PROIEL'], ['UD_Latin-ITTB', 'UD_Latin-PROIEL'], ['UD_Latin-ITTB', 'UD_Latin-Perseus', 'UD_Latin-PROIEL']],
    ['lt', 'Lithuanian', "IE.Baltic", ['UD_Lithuanian-HSE'], ['UD_Lithuanian-HSE'], ['UD_Lithuanian-HSE']],
    ['lv', 'Latvian', "IE.Baltic", ['UD_Latvian-LVTB'], ['UD_Latvian-LVTB'], ['UD_Latvian-LVTB']],
    ['mr', 'Marathi', "IE.Indic", ['UD_Marathi-UFAL'], ['UD_Marathi-UFAL'], ['UD_Marathi-UFAL']],
    ['nl', 'Dutch', "IE.Germanic", ['UD_Dutch-Alpino', 'UD_Dutch-LassySmall'], ['UD_Dutch-Alpino', 'UD_Dutch-LassySmall'], ['UD_Dutch-Alpino', 'UD_Dutch-LassySmall']],
    ['no', 'Norwegian', "IE.Germanic", ['UD_Norwegian-Bokmaal', 'UD_Norwegian-Nynorsk', 'UD_Norwegian-NynorskLIA'], ['UD_Norwegian-Bokmaal', 'UD_Norwegian-Nynorsk'], ['UD_Norwegian-Bokmaal', 'UD_Norwegian-Nynorsk', 'UD_Norwegian-NynorskLIA']],
    ['pl', 'Polish', "IE.Slavic", ['UD_Polish-LFG', 'UD_Polish-SZ'], ['UD_Polish-LFG', 'UD_Polish-SZ'], ['UD_Polish-LFG', 'UD_Polish-SZ']],
    ['pt', 'Portuguese', "IE.Romance", ['UD_Portuguese-Bosque', 'UD_Portuguese-GSD'], ['UD_Portuguese-Bosque', 'UD_Portuguese-GSD'], ['UD_Portuguese-Bosque', 'UD_Portuguese-GSD', 'UD_Portuguese-PUD']],
    ['ro', 'Romanian', "IE.Romance", ['UD_Romanian-RRT', 'UD_Romanian-Nonstandard'], ['UD_Romanian-RRT', 'UD_Romanian-Nonstandard'], ['UD_Romanian-RRT', 'UD_Romanian-Nonstandard']],
    ['ru', 'Russian', "IE.Slavic", ['UD_Russian-SynTagRus', 'UD_Russian-GSD', 'UD_Russian-Taiga'], ['UD_Russian-SynTagRus', 'UD_Russian-GSD'], ['UD_Russian-SynTagRus', 'UD_Russian-GSD', 'UD_Russian-PUD', 'UD_Russian-Taiga']],
    ['sk', 'Slovak', "IE.Slavic", ['UD_Slovak-SNK'], ['UD_Slovak-SNK'], ['UD_Slovak-SNK']],
    ['sl', 'Slovenian', "IE.Slavic", ['UD_Slovenian-SSJ', 'UD_Slovenian-SST'], ['UD_Slovenian-SSJ'], ['UD_Slovenian-SSJ', 'UD_Slovenian-SST']],
    ['sr', 'Serbian', "IE.Slavic", ['UD_Serbian-SET'], ['UD_Serbian-SET'], ['UD_Serbian-SET']],
    ['sv', 'Swedish', "IE.Germanic", ['UD_Swedish-Talbanken', 'UD_Swedish-LinES'], ['UD_Swedish-Talbanken', 'UD_Swedish-LinES'], ['UD_Swedish-Talbanken', 'UD_Swedish-LinES', 'UD_Swedish-PUD']],
    ['ta', 'Tamil', "Dravidian.Southern", ['UD_Tamil-TTB'], ['UD_Tamil-TTB'], ['UD_Tamil-TTB']],
    ['te', 'Telugu', "Dravidian.South-Central", ['UD_Telugu-MTG'], ['UD_Telugu-MTG'], ['UD_Telugu-MTG']],
    # ['th', 'Thai', [], [], ['UD_Thai-PUD']],
    # ['tl', 'Tagalog', [], [], ['UD_Tagalog-TRG']],
    ['tr', 'Turkish', "Turkic.Southwestern", ['UD_Turkish-IMST'], ['UD_Turkish-IMST'], ['UD_Turkish-IMST', 'UD_Turkish-PUD']],
    ['uk', 'Ukrainian', "IE.Slavic", ['UD_Ukrainian-IU'], ['UD_Ukrainian-IU'], ['UD_Ukrainian-IU']],
    ['ur', 'Urdu', "IE.Indic", ['UD_Urdu-UDTB'], ['UD_Urdu-UDTB'], ['UD_Urdu-UDTB']],
    ['vi', 'Vietnamese', "Austro-Asiatic.Viet-Muong", ['UD_Vietnamese-VTB'], ['UD_Vietnamese-VTB'], ['UD_Vietnamese-VTB']],
    ['zh', 'Chinese', "Sino-Tibetan", ['UD_Chinese-GSD'], ['UD_Chinese-GSD'], ['UD_Chinese-GSD', 'UD_Chinese-CFL', 'UD_Chinese-HK', 'UD_Chinese-PUD']],
)

# confs
UD2_DIR = "./ud-treebanks-v2.3/"
OUT_DIR = "./ud23/"
LIB_DIR = "./fastText_multilingual/"

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

# filter extended rows and comments
def deal_conll_file(fin, fout):
    for line in fin:
        line = line.strip()
        fields = line.split("\t")
        if len(line) == 0:
            fout.write("\n")
        else:
            try:
                z = int(fields[0])
                fout.write("\t".join(fields)+"\n")
            except:
                pass

#
def main(extract_first_only):
    # # first get the English one
    # lang = "en"
    # system("wget -nc -O %s/wiki.%s.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.%s.vec" % (OUT_DIR, lang, lang), pp=True)
    # # en_dict = FastVector(vector_file='%s/wiki.en.vec' % OUT_DIR)
    if extract_first_only:
        printing("Extract first-treebank mode!!")
    else:
        printing("Extract all mode!!")
    #
    for zzz in LANGUAGE_LIST:
        lang = zzz[0]
        printing("Dealing (extract-first=%s) with lang %s: %s" % (extract_first_only, lang, zzz))
        train_first_dir = zzz[-3][0]
        for curf, cur_dirs in zip(["train", "dev", "test"], zzz[-3:]):
            # =====
            if extract_first_only:
                cur_dirs = [train_first_dir]
            # =====
            out_fname = "%s/%s_%s.conllu" % (OUT_DIR, lang, curf)
            fout = zopen(out_fname, "w")
            for one_dir in cur_dirs:
                last_name = one_dir.split("-")[-1].lower()
                path_name = "%s/%s/%s_%s-ud-%s.conllu" % (UD2_DIR, one_dir, lang, last_name, curf)
                if os.path.exists(path_name):
                    with zopen(path_name) as fin:
                        deal_conll_file(fin, fout)
                else:
                    assert curf=="dev" and lang in ["kk", "hy"]
                    printing("Warn: no dev, place in test.")
                    with zopen("%s/%s/%s_%s-ud-%s.conllu" % (UD2_DIR, one_dir, lang, last_name, "test")) as fin:
                        deal_conll_file(fin, fout)
            fout.close()
            # stat
            system('cat %s | grep -E "^$" | wc' % out_fname, pp=True)
            system('cat %s | grep -Ev "^$" | wc' % out_fname, pp=True)
            system("cat %s | grep -Ev '^$' | cut -f 5 -d $'\t'| grep -Ev 'PUNCT|SYM' | wc" % out_fname, pp=True)
        # concat for all file
        system("cat %s/%s_train.conllu %s/%s_dev.conllu %s/%s_test.conllu >%s/%s_all.conllu"
               % (OUT_DIR, lang, OUT_DIR, lang, OUT_DIR, lang, OUT_DIR, lang))

def main2():
    for zzz in LANGUAGE_LIST:
        lang = zzz[0]
        # get original embed
        system("wget -nc -O %s/wiki.%s.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.%s.vec" % (OUT_DIR, lang, lang), pp=True)
        # project with LIB-matrix
        lang_dict = FastVector(vector_file='%s/wiki.%s.vec' % (OUT_DIR, lang))
        lang_dict.apply_transform("%s/alignment_matrices/%s.txt" % (LIB_DIR, lang))
        lang_dict.export("%s/wiki.multi.%s.vec" % (OUT_DIR, lang))

if __name__ == '__main__':
    # todo(warn): only extract first train-treebank
    # main(False)
    main(True)
    main2()

# The prepared files are: OUT_DIR/{*_*.conllu, wiki.multi.*.vec}
# python3 zprep_ud23.py |& grep -v "s$" | tee ud23/log


# todo(warn): extras for zh-s/t (simplified or traditional)
"""
ln -s zh_train.conllu zht_train.conllu
ln -s zh_test.conllu zht_test.conllu
ln -s zh_dev.conllu zht_dev.conllu
python3 zconv_aug_emb.py s2t wiki.zh.vec wiki.zht.vec
python3 zconv_aug_emb.py t2s wiki.zh.vec wiki.zhs.vec
python3 zconv_aug_emb.py s2t wiki.multi.zh.vec wiki.multi.zht.vec
python3 zconv_aug_emb.py t2s wiki.multi.zh.vec wiki.multi.zhs.vec
"""

"""
*: notes about the parsing data preparation:
For PTB and CTB, following conventions. These are written previously and already in the conllu format (POS at 3), thus directly copy previous data from mercury.
For UD, update to UDv2.3 and prepare data from it, currently only extract one treebank per language, which is different from the previous MDP project (zprep_ud22).
"""
