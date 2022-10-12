#

# split fulltext docs

import sys
import json
from msp2.utils import zlog
from msp2.data.rw import DataReader, LineStreamer, DataWriter, get_text_dumper

# from open-sesame
TEST_FILES = {
    "ANC__110CYL067.xml",
    "ANC__110CYL069.xml",
    "ANC__112C-L013.xml",
    "ANC__IntroHongKong.xml",
    "ANC__StephanopoulosCrimes.xml",
    "ANC__WhereToHongKong.xml",
    "KBEval__atm.xml",
    "KBEval__Brandeis.xml",
    "KBEval__cycorp.xml",
    "KBEval__parc.xml",
    "KBEval__Stanford.xml",
    "KBEval__utd-icsi.xml",
    "LUCorpus-v0.3__20000410_nyt-NEW.xml",
    "LUCorpus-v0.3__AFGP-2002-602187-Trans.xml",
    "LUCorpus-v0.3__enron-thread-159550.xml",
    "LUCorpus-v0.3__IZ-060316-01-Trans-1.xml",
    "LUCorpus-v0.3__SNO-525.xml",
    "LUCorpus-v0.3__sw2025-ms98-a-trans.ascii-1-NEW.xml",
    "Miscellaneous__Hound-Ch14.xml",
    "Miscellaneous__SadatAssassination.xml",
    "NTI__NorthKorea_Introduction.xml",
    "NTI__Syria_NuclearOverview.xml",
    "PropBank__AetnaLifeAndCasualty.xml",
}

# those not in the partial list of "GeneralReleaseNotes1.5.pdf"
NonPartialTest_FILES = {
    "ANC__112C-L013.xml",
    "ANC__StephanopoulosCrimes.xml",
    "KBEval__atm.xml",
    "KBEval__utd-icsi.xml",
    "LUCorpus-v0.3__20000410_nyt-NEW.xml",
    "LUCorpus-v0.3__AFGP-2002-602187-Trans.xml",
    "LUCorpus-v0.3__SNO-525.xml",
    "LUCorpus-v0.3__sw2025-ms98-a-trans.ascii-1-NEW.xml",
    "NTI__NorthKorea_Introduction.xml",
    "NTI__Syria_NuclearOverview.xml",
    "PropBank__AetnaLifeAndCasualty.xml",
}

DEV_FILES = {
    "ANC__110CYL072.xml",
    "KBEval__MIT.xml",
    "LUCorpus-v0.3__20000415_apw_eng-NEW.xml",
    "LUCorpus-v0.3__ENRON-pearson-email-25jul02.xml",
    "Miscellaneous__Hijack.xml",
    "NTI__NorthKorea_NuclearOverview.xml",
    "NTI__WMDNews_062606.xml",
    "PropBank__TicketSplitting.xml",
}

# todo(note): we only include files that appear in FN1.5 since we do not know whether partial or not in Fn1.7
TRAIN_FILES = ['ANC__110CYL068', 'ANC__110CYL070', 'ANC__110CYL200', 'ANC__EntrepreneurAsMadonna', 'ANC__HistoryOfGreece', 'ANC__HistoryOfJerusalem', 'ANC__HistoryOfLasVegas', 'ANC__IntroJamaica', 'ANC__IntroOfDublin', 'C-4__C-4Text', 'KBEval__LCC-M', 'KBEval__lcch', 'LUCorpus-v0.3__20000416_xin_eng-NEW', 'LUCorpus-v0.3__20000419_apw_eng-NEW', 'LUCorpus-v0.3__20000420_xin_eng-NEW', 'LUCorpus-v0.3__20000424_nyt-NEW', 'LUCorpus-v0.3__602CZL285-1', 'LUCorpus-v0.3__AFGP-2002-600002-Trans', 'LUCorpus-v0.3__AFGP-2002-600045-Trans', 'LUCorpus-v0.3__CNN_AARONBROWN_ENG_20051101_215800.partial-NEW', 'LUCorpus-v0.3__CNN_ENG_20030614_173123.4-NEW-1', 'LUCorpus-v0.3__artb_004_A1_E1_NEW', 'LUCorpus-v0.3__artb_004_A1_E2_NEW', 'LUCorpus-v0.3__wsj_1640.mrg-NEW', 'LUCorpus-v0.3__wsj_2465', 'NTI__BWTutorial_chapter1', 'NTI__ChinaOverview', 'NTI__Iran_Biological', 'NTI__Iran_Chemical', 'NTI__Iran_Introduction', 'NTI__Iran_Missile', 'NTI__Iran_Nuclear', 'NTI__Kazakhstan', 'NTI__LibyaCountry1', 'NTI__NorthKorea_ChemicalOverview', 'NTI__NorthKorea_NuclearCapabilities', 'NTI__Russia_Introduction', 'NTI__SouthAfrica_Introduction', 'NTI__Taiwan_Introduction', 'NTI__WMDNews_042106', 'NTI__workAdvances', 'PropBank__BellRinging', 'PropBank__ElectionVictory', 'PropBank__LomaPrieta', 'PropBank__PolemicProgressiveEducation', 'QA__IranRelatedQuestions', 'SemAnno__Text1']
TRAIN_FILES += ["Miscellaneous__C-4Text", "Miscellaneous__IranRelatedQuestions", "Miscellaneous__SemAnno_1"]  # some change name
TRAIN_FILES = set([x+".xml" for x in TRAIN_FILES])

# # todo(note): some fn17 files annotate too little, but currently do not exclude these.
# # but no those in TEST/DEV to make them consistent with previous works!
# EXCLUDE_FILES = {
#     "ANC__112C-L012",  # no annotations in fn17
#     "ANC__WhatToHongKong",  # ann too few
#     "ANC__chapter1_911report",  # ann too few
#     "ANC__chapter8_911report",  # no annotations in fn17
#     "ANC__journal_christine",  # ann too few
#     "LUCorpus-v0.3__AFGP-2002-600175-Trans",  # no annotations in fn17
#     "WikiTexts__acquisition.n",  # ann too few
#     "WikiTexts__boutique.n",  # ann too few
#     "WikiTexts__extent.n",  # ann too few
#     "WikiTexts__fund.n",  # ann too few
#     "WikiTexts__invoice.n",  # ann too few
#     "WikiTexts__oven.n",  # ann too few
#     "WikiTexts__someone.n",  # ann too few
#     "WikiTexts__spatula.n",  # ann too few
# }
# assert len(EXCLUDE_FILES.intersection(TEST_FILES)) == 0
# assert len(EXCLUDE_FILES.intersection(DEV_FILES)) == 0
# # --
EXCLUDE_FILES = {}

# --
def main(input_file: str, output_prefix: str):
    reader = DataReader(LineStreamer(input_file), "zjson_doc")
    all_docs = {"train": [], "dev": [], "test": [], "test1": [], "exclude": [], "others": []}
    for one_doc in reader:
        check_name = one_doc.id + ".xml"
        # --
        # special test set!
        if check_name in NonPartialTest_FILES:
            assert check_name in TEST_FILES
            all_docs["test1"].append(one_doc)
        # --
        if check_name in TEST_FILES:
            part = "test"
        elif check_name in DEV_FILES:
            part = "dev"
        elif check_name in EXCLUDE_FILES:
            part = "exclude"
        elif check_name in TRAIN_FILES:
            part = "train"
        else:
            part = "others"
        all_docs[part].append(one_doc)
    # write all
    for k, docs in all_docs.items():
        f = f"{output_prefix}.{k}.json"
        zlog(f"Write num={len(docs)} to {f}")
        with DataWriter(get_text_dumper(f)) as writer:
            writer.write_insts(docs)

if __name__ == '__main__':
    main(*sys.argv[1:])

"""
PYTHONPATH=../../src/ python3 fn_split_ft.py fn15/fulltext.json fn15/fulltext
Write num=47 to fn15/fulltext.train.json
Write num=8 to fn15/fulltext.dev.json
Write num=23 to fn15/fulltext.test.json
Write num=11 to fn15/fulltext.test1.json
Write num=0 to fn15/fulltext.exclude.json
Write num=0 to fn15/fulltext.others.json
PYTHONPATH=../../src/ python3 fn_split_ft.py fn17/fulltext.json fn17/fulltext
Write num=47 to fn17/fulltext.train.json
Write num=8 to fn17/fulltext.dev.json
Write num=23 to fn17/fulltext.test.json
Write num=11 to fn17/fulltext.test1.json
Write num=0 to fn17/fulltext.exclude.json
Write num=29 to fn17/fulltext.others.json
"""
