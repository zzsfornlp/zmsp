#

# lexical/frame constrainer
# -- constrains 1) from LU to Frame, 2) from Frame to FE

__all__ = [
    "FN2UD_POS_MAP", "UD2FN_POS_MAP",
    "ConstrainerConf", "Constrainer", "LexConstrainerConf", "LexConstrainer", "FEConstrainerConf", "FEConstrainer",
]

from typing import List, Dict
from msp2.data.inst import Token, Sent, HeadFinder, yield_sents
from msp2.data.rw import ReaderGetterConf
from msp2.utils import Conf, zlog, init_everything, default_json_serializer, JsonSerializable, ZObject

# =====
# common

# speical POS mappings
FN2UD_POS_MAP = {"v": "VERB", "n": "NOUN", "a": "ADJ", "prep": "ADP", "adv": "ADV", "num": "NUM",
                 "c": "CCONJ", "art": "DET", "scon": "SCONJ", "intj": "INTJ", "pron": "PRON"}
UD2FN_POS_MAP = {v:k for k,v in FN2UD_POS_MAP.items()}
# --

class ConstrainerConf(Conf):
    def __init__(self):
        pass

class Constrainer:
    def __init__(self, conf: ConstrainerConf):
        self.conf = conf
        self.cmap = {}  # key -> {value -> Count}
        # --

    def reinit(self):
        raise NotImplementedError()

    def save(self, f):
        default_json_serializer.to_file((self.conf.to_json(), self.cmap), f)

    @classmethod
    def load_from_file(cls, fname: str):
        x = cls(None)
        x.load(fname)
        return x

    def load(self, f):
        conf, cmap = default_json_serializer.from_file(f)
        self.conf.from_json(conf)
        self.cmap = cmap
        self.reinit()
        zlog(f"Load {self.__class__.__name__} from {f}")

    def add(self, key, value, c=1):
        vv = self.cmap.get(key)
        if vv is None:
            vv = {}
            self.cmap[key] = vv
        vv[value] = vv.get(value, 0) + c

    def summary(self):
        m = self.cmap
        return {"numK": len(m), "numK1": len([1 for k,v in m.items() if len(v)==1]),
                "numV": sum(len(z) for z in m.values()),
                "numC": sum(z2 for z in m.values() for z2 in z.values())}

    def get(self, key, df=None):
        return self.cmap.get(key, df)

# =====

class LexConstrainerConf(ConstrainerConf):
    def __init__(self):
        super().__init__()
        # --
        self.cons_ftag = "evt"
        self.lex_feat_f = "lambda t: f'{t.lemma.lower()}|{t.upos}'"  # feature for keyer
        self.use_fn_style = True  # lemma.pos, directly using FN's ones

class LexConstrainer(Constrainer):
    def __init__(self, conf: LexConstrainerConf=None):
        super().__init__(LexConstrainerConf() if conf is None else conf)
        conf: LexConstrainerConf = self.conf
        # --
        self.reinit()  # initialize

    def reinit(self):
        conf = self.conf
        self.lex_feat_f = eval(conf.lex_feat_f)  # lambda Token: Feature
        self.hf = HeadFinder({"ef": "NOUN", "evt": "VERB"}.get(conf.cons_ftag))

    @staticmethod
    def norm_lu(lu: str):  # normalize some LU
        lu_lemma, lu_pos = lu.split(".")
        if "_" in lu_lemma:  # get rid of special ones and split them apart
            lemmas = [z for z in lu_lemma.split("_") if not (len(z)>0 and z[0]=='(' and z[-1]==')')]
            lemma_real = " ".join(lemmas)
        else:
            lemma_real = lu_lemma
        return f"{lemma_real}.{lu_pos}"

    def span2feat(self, sent: Sent, widx: int, wlen: int):  # from a span to feat
        # def span2feat(self, sent: Sent, widx: int, wlen: int, try_head=True):  # from a span to feat
        conf: LexConstrainerConf = self.conf
        hwidx = self.hf.find_shead(sent, widx, wlen)  # try to find head word
        if conf.use_fn_style:
            hpos = sent.seq_upos.vals[hwidx]
            lu_name = " ".join(sent.seq_lemma.vals[widx:widx+wlen]).lower() + "." \
                      + UD2FN_POS_MAP.get(hpos, hpos.lower())
            feat = self.lu2feat(lu_name)
        else:
            tokens = sent.get_tokens(widx, widx+wlen)
            feat = " ".join([self.lex_feat_f(t) for t in tokens])  # my own feat!
        # special try_head if not found
        # if try_head and wlen>0 and feat not in self.cmap:
        #     return self.span2feat(sent, hwidx, 1, False)
        return feat

    def lu2feat(self, lu_name: str):  # from fn styled LU to feat
        lu_lemma, lu_pos = lu_name.rsplit(".", 1)
        t = ZObject(lemma=lu_lemma, upos=FN2UD_POS_MAP.get(lu_pos, "X"))
        return self.lex_feat_f(t)

# =====

class FEConstrainerConf(ConstrainerConf):
    def __init__(self):
        super().__init__()
        # --

class FEConstrainer(Constrainer):
    def __init__(self, conf: FEConstrainerConf=None):
        super().__init__(FEConstrainerConf() if conf is None else conf)
        # --

    def reinit(self):
        pass

# =====
# get the constraints either from lexicon or from data!

class MainConf(Conf):
    def __init__(self):
        self.frame_file = ""
        self.train = ReaderGetterConf()
        self.lex_conf = LexConstrainerConf()
        self.fe_conf = FEConstrainerConf()
        self.lex_save_name = "cons_lex.json"
        self.fe_save_name = "cons_fe.json"

def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    # --
    cons_lex = LexConstrainer(conf.lex_conf)
    cons_fe = FEConstrainer(conf.fe_conf)
    # --
    # some confs
    lex_use_fn_style = conf.lex_conf.use_fn_style
    # --
    # first try to read frame file
    if conf.frame_file:
        assert lex_use_fn_style, "Otherwise do not provide 'frame_file'!!s"
        external_frames = default_json_serializer.from_file(conf.frame_file)
        for fname, fv in external_frames.items():
            # LU
            for lu in fv["lexUnit"]:
                lu_name = lu["name"]
                cons_lex.add(cons_lex.lu2feat(lu_name), fname, c=0)  # no count now, only add entry
                lu_name2 = LexConstrainer.norm_lu(lu_name)
                if lu_name2 != lu_name:  # also add normed name!
                    cons_lex.add(cons_lex.lu2feat(lu_name2), fname, c=0)
            # FE
            for fe in fv["FE"]:
                fe_name = fe["name"]
                cons_fe.add(fname, fe_name, c=0)  # again no count here!
        zlog(f"Read from {conf.frame_file}: LU={cons_lex.summary()}, FE={cons_fe.summary()}")
    # --
    # then read data!
    if conf.train.input_path:
        reader = conf.train.get_reader()
        for sent in yield_sents(reader):
            for frame in sent.get_frames(conf.lex_conf.cons_ftag):
                frame_name = frame.type
                # LU
                feats = []
                if lex_use_fn_style:  # then directly use the stored one!!
                    lu_name = frame.info.get("luName")
                    feats.append(cons_lex.lu2feat(lu_name))
                    lu_name2 = LexConstrainer.norm_lu(lu_name)
                    if lu_name2 != lu_name:
                        feats.append(cons_lex.lu2feat(lu_name2))
                # also add the plain one!!
                widx, wlen = frame.mention.get_span()
                feat = cons_lex.span2feat(frame.sent, widx, wlen)
                feats.append(feat)
                # --
                for feat in feats:
                    cons_lex.add(feat, frame_name, c=1)
                # FE
                for alink in frame.args:
                    cons_fe.add(frame_name, alink.role, c=1)
        zlog(f"Read from {conf.train.input_path}: LU={cons_lex.summary()}, FE={cons_fe.summary()}")
    # --
    # summary and save
    cons_lex.save(conf.lex_save_name)
    cons_fe.save(conf.fe_save_name)

# --
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

"""
# to collect the constraints (at 'run_voc' dir)
# --
# cons0: only frames
PYTHONPATH=../src/ python3 -m msp2.tasks.zsfp.extract.constrainer frame_file:../fn_parsed/fn15_frames.json train.input_path: use_fn_style:1 lex_save_name:cons_lex0.json fe_save_name:cons_fe0.json log_file:_log.cons0
# cons1: frame+data
PYTHONPATH=../src/ python3 -m msp2.tasks.zsfp.extract.constrainer frame_file:../fn_parsed/fn15_frames.json train.input_path:fn15_et_combined.json use_fn_style:1 lex_save_name:cons_lex1.json fe_save_name:cons_fe1.json log_file:_log.cons1
# cons2: only data
PYTHONPATH=../src/ python3 -m msp2.tasks.zsfp.extract.constrainer frame_file: train.input_path:fn15_et_combined.json use_fn_style:1 lex_save_name:cons_lex2.json fe_save_name:cons_fe2.json log_file:_log.cons2
# --
# cons3: 
PYTHONPATH=../src/ python3 -m msp2.tasks.zsfp.extract.constrainer frame_file:../fn_parsed/fn15_frames.json train.input_path: use_fn_style:1 lex_save_name:cons_lex3.json fe_save_name:cons_fe3.json log_file:_log.cons3 "lex_feat_f:lambda t: f'{t.lemma.lower()}|{t.upos}'"
# => see "exp/go_rule_lab.sh" for more ...
"""
