#

# --
# different people have different tokenization methods:
# dygiepp: spacy, extend chars to outside token
# oneie: nltk/self, first chunk by span boundaries and then tokenize

from typing import Set, List, Dict
import os
import json
import argparse
import re
import sys
from collections import Counter

# =====
# helpers
def zlog(s):
    # print(str(s), file=sys.stderr, flush=True)
    print(str(s), file=sys.stdout, flush=True)

def zlog2(s):
    print(str(s), file=sys.stderr, flush=True)

def zopen(filename, mode='r', encoding="utf-8"):
    if 'b' in mode:
        encoding = None
    return open(filename, mode, encoding=encoding)

def zwarn(s):
    # breakpoint()
    zlog("Warning: " + str(s))

#
# normalize special char: by observing and guessing ...
SPECIAL_CHAR_TRANSFER_MAP = {
    "\x85": " ", "\x91": " ", "\x96": " ", "\x97": " ", "\xa0": " ",
    # "\x92": "\'", "´": "\'", "‘": "\'", "’": "\'",
    "\x93": "\"", "\x94": "\"", "“": "\"", "”": "\"",
    "\x92": "\"", "´": "\"", "‘": "\"", "’": "\"",
    # (this one is strange!) "'": "\"",
    "\x7f": " ", "\x8d": " ",
}
def normalize_text(text):
    return "".join(SPECIAL_CHAR_TRANSFER_MAP.get(t, t) for t in text)

# =====
# first strip and extract raw text from xml for later tokenization
# replacing the <> tags with " " or "\n" (indicting sentence splitting or not), return the same length str as input.
# space_names/newline_names: replace only tag fields with space or newline; ignore_names: replace whole contents with space
def strip_xml(orig_str: str, names: dict, strict=True):
    space_names, newline_names, ignore_names = [names[z] for z in ["space", "newline", "ignore"]]
    ret = []  # returns that keep the same chars
    ret_notag = []  # returns that remove all tags
    # --
    is_cur_ignoring = False  # current is ignoring things
    stack = [("ROOT", False)]
    repl_map = {k: " " for k in space_names}
    repl_map.update({k: "\n" for k in newline_names})
    for tok in stream_xml(orig_str):
        if tok[0] == '<':
            assert tok[-1] == ">"
            inner_tok = tok[1:-1]
            is_close, self_close = False, False
            assert len(inner_tok)>0
            if inner_tok[0] == "/":
                inner_tok = inner_tok[1:]
                is_close = True
            assert len(inner_tok)>0
            if inner_tok[-1] == "/":
                inner_tok = inner_tok[:-1]
                assert not is_close
                self_close = True
            assert len(inner_tok)>0
            tag = inner_tok.split()[0].lower()
            # tag
            if tag[0] in "!?":
                # ignore special ones
                ret.append(" " * len(tok))
            else:
                if strict:
                    assert tag in repl_map
                repl = repl_map.get(tag, " ")
                ret.append(repl + " " * (len(tok)-2) + repl)
                # deal with the tags
                if not self_close:
                    if is_close:
                        while True:
                            prev_one, prev_ignore = stack.pop()
                            if prev_one != tag:
                                zwarn(f"Unmatched tag: {prev_one} vs {tag}, "
                                      f"ignore this since there are some errors in data.")
                            else:
                                break
                        is_cur_ignoring = stack[-1][-1]  # reverse previous is_ignore
                    else:
                        new_is_ignore = is_cur_ignoring or (tag in ignore_names)
                        stack.append((tag, new_is_ignore))
                        is_cur_ignoring = new_is_ignore
        else:
            if is_cur_ignoring:
                ret.append(" " * len(tok))
                ret_notag.append(" " * len(tok))
            else:
                ret.append(tok)
                ret_notag.append(tok)
    # --
    ret0, ret1 = "".join(ret), "".join(ret_notag)
    assert len(ret1) == len(re.compile('<.*?>', re.DOTALL).sub("", orig_str))  # sanity check
    return ret0, ret1

# helper function for xml strip, return stream of either raw text or <tag>
# this is actual the tokenizer
def stream_xml(orig_str: str):
    i = 0
    all_len = len(orig_str)
    while i<all_len:
        c = orig_str[i]
        ret = []
        if c == "<":
            # until '>'
            while orig_str[i] != '>':
                ret.append(orig_str[i])
                i += 1
            # eat extra one
            ret.append(orig_str[i])
            i += 1
        else:
            # until '<'
            while i<all_len and orig_str[i] != '<':
                ret.append(orig_str[i])
                i += 1
        yield "".join(ret)
# --

# --
# tokenizer
class BaseTokenizer:
    def __init__(self, lang: str, args):
        self.lang = lang
        self.is_zh = (self.lang == "zh")
        # --

    # str -> List[{"tokens": List[str], "positions": List[(cidx, clen)]}]
    def tokenize(self, input_str: str, offset: int):
        if self.is_zh:
            # note: special handling for zh: tokenize with no space!!
            map_dense2orig = []
            dense_str = []
            for ii, cc in enumerate(input_str):
                if not str.isspace(cc):
                    map_dense2orig.append(ii)
                    dense_str.append(cc)
            # parse dense str first
            dense_str = ''.join(dense_str)
            dense_sents = self._do_tokenize(dense_str, 0)
            # then finally set position
            for sent in dense_sents:
                new_positions = []
                for cidx, clen in sent["positions"]:
                    new_cidx0, new_cidx1 = map_dense2orig[cidx], map_dense2orig[cidx+clen-1]+1
                    new_positions.append((new_cidx0+offset, new_cidx1-new_cidx0))
                sent["positions"] = new_positions
            return dense_sents
        else:
            return self._do_tokenize(input_str, offset)

    def _do_tokenize(self, input_str: str, offset: int):
        raise NotImplementedError()

    @staticmethod
    def get_tokenizer(which: str, lang: str, args):
        if which == "stanza": return StanzaTokenizer(lang, args)
        elif which == "corenlp": return CorenlpTokenizer(lang, args)
        else: raise NotImplementedError(f"UNK tokenizer: {which}")

# prepare
"""
-- use stanza
pip install stanza
[stanza.download(z) for z in ['en','zh','es','ar']]
"""
class StanzaTokenizer(BaseTokenizer):
    def __init__(self, lang: str, args):
        super().__init__(lang, args)
        import stanza
        self.nlp = stanza.Pipeline(processors='tokenize', lang=lang)

    def _do_tokenize(self, input_str: str, offset: int):
        res = self.nlp(input_str)
        rets = []
        for sent in res.sentences:
            tokens, positions = [], []
            for tok in sent.tokens:
                _text, _start_char, _end_char = tok.text, tok._start_char, tok._end_char
                assert _text == input_str[_start_char:_end_char]
                tokens.append(_text)
                positions.append((_start_char+offset, _end_char-_start_char))
            rets.append({"tokens": tokens, "positions": positions})
        return rets

# prepare
"""
-- also use stanza
import stanza
stanza.install_corenlp(dir="YOUR_CORENLP_FOLDER")  # by default ~/stanza_corenlp
[stanza.download_corenlp_models(model=z, version='4.2.0') for z in ["english", "chinese", "spanish", "arabic"]]
"""
class CorenlpTokenizer(BaseTokenizer):
    def __init__(self, lang: str, args):
        super().__init__(lang, args)
        from stanza.server import CoreNLPClient
        self.client = CoreNLPClient(annotators=['tokenize', 'ssplit'],
                                    timeout=60000, memory='16G', properties=lang, be_quite=True)  # be_quite=True

    def __del__(self):
        self.client.stop()

    def _do_tokenize(self, input_str: str, offset: int):
        res = self.client.annotate(input_str)
        rets = []
        for sent in res.sentence:
            tokens, positions = [], []
            for tok in sent.token:
                _text, _start_char, _end_char = tok.originalText, tok.beginChar, tok.endChar
                # special fix for some strange text like emoji ...
                if _end_char - _start_char > len(_text):
                    zwarn(f"Strange token: {_text} [{_start_char}, {_end_char})")
                    _end_char = _start_char + len(_text)
                # --
                assert _text == input_str[_start_char:_end_char]
                tokens.append(_text)
                positions.append((_start_char+offset, _end_char - _start_char))
            rets.append({"tokens": tokens, "positions": positions})
        return rets
# --

# --
# char indexer: from char indexes to token indexes
class CharIndexer:
    def __init__(self, full_char_idxes: List, offset_str: str, sents: List[Dict]):
        self.full_char_idxes = full_char_idxes
        self.offset_str = offset_str
        self.sents = sents

    @staticmethod
    def build_char_indexer(offset_str: str, sents: List[Dict]):
        full_char_idxes = [None] * len(offset_str)
        #
        for sid, one_sent in enumerate(sents):
            assert len(one_sent["tokens"]) == len(one_sent["positions"])
            wid = 0
            for one_tok, one_posi in zip(one_sent["tokens"], one_sent["positions"]):
                cstart, clen = one_posi
                assert ''.join(one_tok.split()) == ''.join(offset_str[cstart:cstart+clen].split())
                for cc in range(cstart, cstart+clen):
                    assert full_char_idxes[cc] is None  # None means blank!!
                    full_char_idxes[cc] = (sid, wid)  # (sid, wid)
                wid += 1
        return CharIndexer(full_char_idxes, offset_str, sents)

    # (cidx, clen) -> List[(sid, wid)]
    def collect_tokens(self, char_idx: int, char_len: int):
        # collect all tokens
        index_chars = self.full_char_idxes
        tokens = []
        for ii in range(char_idx, char_idx+char_len):
            vv = index_chars[ii]
            if vv is not None:
                if len(tokens) == 0 or vv != tokens[-1]:  # find a new one
                    assert len(tokens)==0 or (vv[0]==tokens[-1][0] and vv[1]==tokens[-1][1]+1) \
                           or (vv[0]==tokens[-1][0]+1 and vv[1]==0)  # assert continuing span
                    tokens.append(vv)
        # --
        # check
        str0 = ''.join(self.offset_str[char_idx:char_idx+char_len].split())
        str1 = ''.join([''.join(self.sents[sid]["tokens"][wid].split()) for sid,wid in tokens])
        if str0 not in str1:
            # note: a very strange 'ar' case ...
            if str1 == ''.join(str0.split("_")) or set(str0).difference(set(str1))==set(chr(1618)):
                zwarn(f"=> Slightly unmatch: {str0} vs {str1}")
            else:
                raise RuntimeError()
        return tokens

    # (cidx, clen) -> (sid, widx, wlen)
    # -- hint_sidx helps to select the sentence if split-sent (mostly from head_posi)
    def find_token_posi(self, char_idx: int, char_len: int, hint_sid=None):
        assert char_len > 0
        positions = self.collect_tokens(char_idx, char_len)  # List[sid, widx]
        # check it and report error?
        ret_code = ""
        if len(positions) == 0:
            ret_code = "ErrNotFound"  # no non-blank words at those position (maybe annotations in tag or ignored tag)
            ret_posi = None
        else:
            if len(set([z[0] for z in positions])) > 1:  # more than 1 sentence
                hint_positions = [z for z in positions if z[0]==hint_sid]
                if len(hint_positions) == 0:
                    ret_code = "ErrDiffSent"  # sent splitting error
                    ret_posi = None
                else:
                    ret_code = "WarnDiffSent"  # accept this part
                    ret_posi = (hint_positions[0][0], hint_positions[0][1], len(hint_positions))
            else:
                ret_posi = (positions[0][0], positions[0][1], len(positions))  # (sid, widx, wlen)
                # check boundaries
                ret_cidx = self.sents[ret_posi[0]]["positions"][ret_posi[1]][0]
                ret_cridx = sum(self.sents[ret_posi[0]]["positions"][ret_posi[1]+ret_posi[2]-1])
                # inside or the more ones should be empty
                assert all(str.isspace(c) for c in self.offset_str[char_idx:ret_cidx]+self.offset_str[ret_cridx:char_idx+char_len])
                if ret_cidx < char_idx:
                    ret_code += "WarnLeft"
                if ret_cridx > (char_idx+char_len):
                    if ret_cridx-(char_idx+char_len)==1 and self.offset_str[ret_cridx-1]==".":  # simply a dot
                        ret_code += "WarnRDot"
                    else:
                        ret_code += "WarnRight"
        # --
        # print the mismatches
        if ret_posi is None:
            zlog2(f"=> Cannot find span: {self.offset_str[char_idx:char_idx+char_len]}")
        elif ret_code != "":
            str0 = ' '.join(self.offset_str[char_idx:char_idx+char_len].split())
            str1 = ' '.join(self.sents[ret_posi[0]]["tokens"][ret_posi[1]:ret_posi[1]+ret_posi[2]])
            zlog2(f"=> Span mismatch ({ret_code}): {str0} ||| {str1}")
        # --
        return ret_posi, ret_code
# --

# --
class MainDriver:
    # --
    _SPECIL_TAG_NAMES = {
        "ace": {
            "space": {"quote", },  # 'quote' has no effects on ACE since quote are all in tags
            "newline": {"doc", "docid", "doctype", "datetime", "body", "headline", "text", "turn", "speaker",
                        "endtime", "post", "poster", "postdate", "subject"},
            "ignore": {"docid", "doctype", "datetime", "speaker", "endtime", "postdate"},
        },
        "ere": {
            "space": {"a", "img"},
            "newline": {"doc", "docid", "doctype", "datetime", "body", "headline", "text", "p", "post", "quote",
                        "author", "date_time", "dateline", "keyword"},
            "ignore": {"docid", "doctype", "datetime", "author", "data_tme", "dateline", "img",
                       # note: some ere datasets do but some do not, for simplicity,
                       #  -> we ignore all quotes since they repeat even though have annotations!!
                       "quote",
                       },
        },  # kbp15 is the same!
    }
    # --

    def __init__(self, lang: str, dset: str, args):
        self.lang = lang
        self.dset = dset
        self.args = args
        # currently only support these
        assert self.lang in ["en", "zh", "es", "ar"]
        assert self.dset in ["ace", "ere", "kbp15", "plain"]
        # tokenizer
        self.tokenizer = BaseTokenizer.get_tokenizer(args.tokenizer, lang, args)
        # --

    # modify things inplace!
    def process(self, doc: dict, stat: Counter):
        # =====
        # part 1: pre-processing
        # --
        # first get source string and offset string
        orig_str = normalize_text(doc["text"])
        # --
        # special fixing for str: some of the ERE data miss '>' for the ending '</DOC>'
        if orig_str.endswith("</DOC\n"):
            zwarn(f"Fix missing '>' for doc: {doc['id']}")
            orig_str = orig_str + ">"
        # --
        # blank out certain parts specifically for different datasets!
        if self.dset == "ace":
            orig_notag_str = re.compile('<.*?>', re.DOTALL).sub("", orig_str)
            tok_str, offset_str = strip_xml(orig_str, MainDriver._SPECIL_TAG_NAMES["ace"], True)
        elif self.dset in ["ere", "kbp15"]:
            tok_str, _ = strip_xml(orig_str, MainDriver._SPECIL_TAG_NAMES["ere"], True)
            offset_str = tok_str  # offset is the same as tok!
        else:
            tok_str = offset_str = orig_str
        # --
        # =====
        # part 2: tokenize and locate all the tokens and build char index!
        # --
        if self.args.do_step_tok:
            sents = []
            cur_offset = 0
            # tokenization according to (blanked) tok_str; # todo(note): first tokenize by "\n\n"
            for one_piece_source in tok_str.split("\n\n"):
                one_piece_source = one_piece_source.strip()
                if len(one_piece_source) == 0:
                    continue  # ignore empty ones
                one_offset = offset_str.index(one_piece_source, cur_offset)  # start of this piece
                cur_offset = one_offset + len(one_piece_source)
                # change newline into blank
                one_piece_source = "".join([" " if z == "\n" else z for z in one_piece_source])
                sents.extend(self.tokenizer.tokenize(one_piece_source, one_offset))
            doc["sents"] = sents
        # --
        # =====
        # part 3: get token positions for all spans
        if self.args.do_step_set:
            # build char index!
            char_indexer = CharIndexer.build_char_indexer(offset_str, doc["sents"])
            # go!!
            failed_ids = {z: set() for z in ["ef", "rel", "evt"]}
            for kk in ["ef", "rel", "evt"]:
                # first get positions for all mentions
                for mention in doc[f"{kk}_mentions"]:
                    # resolve posi
                    posi = mention["posi"]
                    if posi is None:
                        assert kk == 'rel' and self.dset == "ere"
                        continue
                    # first for head if there are
                    hint_sid = None
                    if "head" in posi:
                        head_posi, head_code = char_indexer.find_token_posi(
                            posi["head"]["posi_char"][0], posi["head"]["posi_char"][1])
                        if head_posi is None:
                            del posi["head"]  # for head, simply delete this one!!
                        else:
                            posi["head"]["posi_token"] = head_posi
                            posi["head"]["posi_token_code"] = head_code
                            hint_sid = head_posi[0]
                        stat[f"{kk}_head_{head_code}"] += 1
                    else:
                        stat[f"{kk}_head0"] += 1
                    # then for the full one
                    span_posi, span_code = char_indexer.find_token_posi(posi["posi_char"][0], posi["posi_char"][1], hint_sid)
                    stat[f"{kk}_span_{span_code}"] += 1
                    posi["posi_token"] = span_posi
                    posi["posi_token_code"] = span_code
                    if kk != "rel" and span_posi is None:  # note: no span failing for rel!!
                        failed_ids[kk].add(mention["id"])
                # delete mention
                if kk != "rel":
                    cur_failed_ids = failed_ids[kk]
                    doc[f"{kk}_mentions"] = sorted([z for z in doc[f"{kk}_mentions"] if z['id'] not in cur_failed_ids],
                                                   key=lambda x: x["posi"]["posi_token"])
            # --
            # further check failed ef as arg
            failed_ef_ids = failed_ids["ef"]
            for kk in ["rel", "evt"]:
                # first delete args
                for mention in doc[f"{kk}_mentions"]:
                    # --
                    if kk == "rel":  # note: simple checking, ace relation can have Time-* arg
                        assert len([z for z in mention["args"] if not z["role"].startswith("Time")]) == 2
                    # --
                    orig_num_arg = len(mention["args"])
                    mention["args"] = [z for z in mention["args"] if z["aid"] not in failed_ef_ids]
                    now_num_arg = len(mention["args"])
                    stat[f"{kk}_arg_del"] += (orig_num_arg - now_num_arg)
                    stat[f"{kk}_arg_keep"] += now_num_arg
                # then check rel who looses core args & delete it
                if kk == "rel":
                    orig_num_rel = len(doc[f"{kk}_mentions"])
                    valid_items = []
                    for mention in doc[f"{kk}_mentions"]:
                        if len([z for z in mention["args"] if not z["role"].startswith("Time")]) == 2:
                            valid_items.append(mention)
                        else:
                            failed_ids[kk].add(mention['id'])
                    doc[f"{kk}_mentions"] = valid_items
                    now_num_rel = len(valid_items)
                    stat[f"{kk}_ac_del"] += (orig_num_rel - now_num_rel)
                    stat[f"{kk}_ac_keep"] += now_num_rel
            # --
            # finally delete for cluster
            for kk in ["ef", "rel", "evt"]:
                # delete for cluster
                cur_failed_ids = failed_ids[kk]
                stat[f"{kk}_zfail"] += len(cur_failed_ids)
                # --
                for cluster in doc[f"{kk}_clusters"]:
                    orig_cluster_ids = cluster["ids"]
                    cluster["ids"] = [z for z in cluster["ids"] if z not in cur_failed_ids]
                    if len(orig_cluster_ids) != 0 and len(cluster["ids"]) == 0:
                        stat[f"{kk}_cluster_del"] += 1
                        # if kk == 'rel': breakpoint()
                doc[f"{kk}_clusters"] = [z for z in doc[f"{kk}_clusters"] if len(z["ids"]) > 0]  # delete empty ones!
                stat[f"{kk}_cluster_keep"] += len(doc[f"{kk}_clusters"])
            # --
        # --
        stat["doc"] += 1
        return doc

# --
def parse_args():
    parser = argparse.ArgumentParser("Tok data.")
    # basics
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-c", "--code", type=str)  # "{lang}.{dset}" if None, then guess from inputname
    # tokenize
    parser.add_argument("--tokenizer", type=str, default="corenlp")
    # easier to debug
    parser.add_argument("--do_step_tok", type=int, default=1)
    parser.add_argument("--do_step_set", type=int, default=1)
    # --
    args = parser.parse_args()
    return args

def guess_code(code: str, input_file: str):
    if code is None:
        lang, dset = os.path.basename(input_file).split(".")[:2]
    else:
        lang, dset = code.split(".")
    return lang, dset

def main():
    # check basic info
    args = parse_args()
    lang, dset = guess_code(args.code, args.input)
    driver = MainDriver(lang, dset, args)
    # go!
    # --
    all_docs = []
    stat = Counter()
    with zopen(args.input) as fd:
        for line in fd:
            doc = json.loads(line)
            driver.process(doc, stat)
            all_docs.append(doc)
    stat_str = '\n'.join(f"-- {k}: {stat[k]}" for k in sorted(stat.keys()))
    zlog(f"#== Process {lang}.{dset} ({args.input}) {len(all_docs)}: stat=\n{stat_str}")
    if args.output:
        with zopen(args.output, 'w') as fd:
            for doc in all_docs:
                fd.write(json.dumps(doc, ensure_ascii=False) + "\n")
    # --

# --
if __name__ == '__main__':
    main()

# --
# python3 s2_tokenize.py -i tmp.json -o tmp2.json -c en.ace
