#

# maybe some pre-specified tables for convenience

# --
# first on what resources are we interested in
"""
FN:
en (1.5, 1.7)
PB:
en (WSJ, onto5, PBv3), zh/ar (onto5)
Evt:
en/zh/ar (ACE05), en/zh/es (RichERE)
UProp:
en-ewt, fi-tdt, fr-gsd, de-gsd, it-isdt, pt-bosque, es-ancora, es-gsd, zh-gsd
UD(UPOS/UDEP):
...
XNLI:
en,fr,es,de // bg,ru,tr,ar,el,vi,hi,ur,zh // th,sw
PAWS-X:
en,fr,es,de // zh,ja,ko
NER-CONLL:
en,de,es,nl
XQuAD:
en,es,de // el,ru,tr,ar,vi,zh,hi // th
TyDiQA-GoldP:
todo(+N)?
"""

# --
# paths:
"""
0. methods: MTL; SEQ-tune; SEQ-adapter;
1. cross-lingual:
1.1: UD1.4+UPB, or UD2.7, or different treebanks?
1.1.5: UD1.4+EWT -> FiPB
1.2: en/ar/zh: ud, onto5, ace; or es and ere?
1.3: further on other tasks, like ner/ie/xnli/xqa/...?
2. cross-formalism:
2.1: ud/pb+fn -> ace/ere/rams/maven or new(few_shot) frames?
"""

# --
__all__ = [
    "DATA_SHORTCUTS", "parse_filename_with_shortcut",
]

# --
# shortcut names for some data files! (2-layer hierarchy)
# note: 0 for train, 1 for dev, 2 for test
# --
# ud1.4
_UD14 = {}
for cl, full_name in [
    ["en", "UD_English"], ["zh", "UD_Chinese-S"], ["fi", "UD_Finnish"], ["fr", "UD_French"], ["de", "UD_German"],
    ["it", "UD_Italian"], ["pt_bosque", "UD_Portuguese-Bosque"], ["es_ancora", "UD_Spanish-AnCora"], ["es", "UD_Spanish"],
    ["fi_ftb", "UD_Finnish-FTB"],
]:
    for ii, wset in enumerate(["train", "dev", "test"]):
        _UD14[f"{cl}{ii}"] = f"ud-treebanks-v1.4/{full_name}/{cl}-ud-{wset}.conllu"
# --
# ud2.7
_UD27 = {}
for cl, full_name in [
    ["en_ewt", "UD_English-EWT"], ["zh_gsdsimp", "UD_Chinese-GSDSimp"], ["fi_tdt", "UD_Finnish-TDT"],
    ["fr_gsd", "UD_French-GSD"], ["de_gsd", "UD_German-GSD"], ["it_isdt", "UD_Italian-ISDT"],
    ["pt_bosque", "UD_Portuguese-Bosque"], ["es_ancora", "UD_Spanish-AnCora"], ["es_gsd", "UD_Spanish-GSD"],
    ["ar_padt", "UD_Arabic-PADT"], ["ar_nyuad", "UD_Arabic-NYUAD"], ["ca_ancora", "UD_Catalan-AnCora"],
]:
    for ii, wset in enumerate(["train", "dev", "test"]):
        _UD27[f"{cl}{ii}"] = f"ud-treebanks-v2.7/{full_name}/{cl}-ud-{wset}.conllu"
# --
# up1.0 + ewt3.1 + fipb
_UP10 = {}
for ii, wset in enumerate(["train", "dev", "test"]):
    for cl, full_name in [
        ["zh", "UP_Chinese-S"], ["fi", "UP_Finnish"], ["fr", "UP_French"], ["de", "UP_German"], ["it", "UP_Italian"],
        ["pt_bosque", "UP_Portuguese-Bosque"], ["es_ancora", "UP_Spanish-AnCora"], ["es", "UP_Spanish"],
        # ["en", "UP_English2-EWT"],  # note: not using this!
        ["es2", "UP_Spanish2"],  # note: simply combining the two spanish ones!
    ]:
        _UP10[f"{cl}{ii}"] = f"UniversalPropositions/{full_name}/{cl}-up-{wset}.json"
    # ewt3.1
    _UP10[f"ewt{ii}"] = f"pb2/en.ewt.{wset}.json"
    # fipb
    _UP10[f"fipb{ii}"] = f"pb2/fipb-ud-{wset}.json"
# --
# pb(conll05/conll12)
_PB = {}
# conll05
for ii, wset in enumerate(["train", "dev", "test.wsj", "test.brown"]):
    _PB[f"pb05{ii}"] = f"pb12/pb05.{wset}.conll.ud.json"
# conll12
for ii, wset in enumerate(["train", "dev", "test"]):
    for cl in ["en", "zh", "ar"]:
        _PB[f"{cl}{ii}"] = f"pb12/{cl}.{wset}.conll.ud.json"
        _PB[f"{cl}2{ii}"] = f"pb12/{cl}.{wset}.conll.ud2.json"
        _PB[f"{cl}3{ii}"] = f"pb12/{cl}.{wset}.conll.ud3.json"
# --
# ee(event extraction)
_EE = {}
for suff in ["", "2", "S"]:  # S is the shuffled version to ee2
    _tmp_ee = {}
    for ii, wset in enumerate(["train", "dev", "test"]):
        # ace
        for cl in ["en", "ar", "zh"]:
            _tmp_ee[f"{cl}_ace{ii}"] = f"data/split{suff}/{cl}.ace.{wset}.json"
        _tmp_ee[f"en2_ace{ii}"] = f"data/split{suff}/en.ace2.{wset}.json"
        # ere
        for cl in ["en", "es", "zh"]:
            _tmp_ee[f"{cl}_ere{ii}"] = f"data/split{suff}/{cl}.ere.{wset}.json"
        if wset == "train":
            _tmp_ee[f"en_kbp15{ii}"] = f"data/split{suff}/en.kbp15.json"
    _EE[suff] = _tmp_ee
# --
# pf(pb/fn) (pb3 and fn15/17)
_PF = {}
for ii, wset in enumerate(["train", "dev", "test"]):
    _PF[f"ewt{ii}"] = f"pbfn/en.ewt.{wset}.ud.json"
    _PF[f"onto{ii}"] = f"pbfn/en.onto.{wset}.ud.json"
    _PF[f"ontoC{ii}"] = f"pbfn/en.ontoC.{wset}.ud.json"  # onto with conll12 split
    _PF[f"fn15{ii}"] = f"pbfn/en.fn15.{wset}.ud3.json"
    _PF[f"fn17{ii}"] = f"pbfn/en.fn17.{wset}.ud3.json"
# extras ones
_PF[f"ontoT"] = f"pbfn/en.onto.conll12-test.ud.json"
_PF[f"fn15E"] = f"pbfn/en.fn15.exemplars.ud3.json"
_PF[f"fn17E"] = f"pbfn/en.fn17.exemplars.ud3.json"
# --
DATA_SHORTCUTS = {
    "ud14": _UD14, "ud27": _UD27, "ud": _UD27, "up10": _UP10, "up": _UP10, "pb": _PB,
    "ee": _EE[""], "ee2": _EE["2"], "eeS": _EE["S"], "pf": _PF,
}

# function
def parse_filename_with_shortcut(s: str):
    s2 = None
    if s.startswith("_"):  # special signature!!
        cur_node = DATA_SHORTCUTS
        for f in s[1:].split('/'):
            cur_node = cur_node.get(f)
            if cur_node is None:
                break
        if isinstance(cur_node, str):
            s2 = cur_node
    return s2
