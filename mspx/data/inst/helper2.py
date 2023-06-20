#

__all__ = [
    'yield_sents', 'yield_frames', 'yield_pairs', 'yield_sent_pairs',
    'NLTKTokenizer', 'get_label_gs',
]

from mspx.utils import zwarn

# --
# data yielder

# yield sents
def yield_sents(insts):
    from .doc import Doc, Sent
    from .frame import Frame
    # --
    if isinstance(insts, (Doc, Sent, Frame)):
        insts = [insts]
    for inst in insts:
        if isinstance(inst, Doc):
            yield from inst.sents
        elif isinstance(inst, Sent):
            yield inst
        elif isinstance(inst, Frame):
            yield inst.sent  # go up!!
        else:
            raise NotImplementedError(f"Error: bad input {type(inst)}")

# yield frames!
def yield_frames(insts, sent_level=False, *args, **kwargs):
    from .doc import Doc, Sent
    from .frame import Frame
    # --
    if isinstance(insts, (Doc, Sent, Frame)):
        insts = [insts]
    for inst in insts:
        if isinstance(inst, Doc):
            if sent_level:  # make sure inside sents!
                for sent in inst.sents:
                    yield from sent.yield_frames(*args, **kwargs)
            else:  # allow doc-level!
                yield from inst.yield_frames(*args, **kwargs)
        elif isinstance(inst, Sent):
            yield from inst.yield_frames(*args, **kwargs)
        elif isinstance(inst, Frame):
            yield inst
        else:
            raise NotImplementedError(f"Error: bad input {type(inst)}")
    # --

# yield pairs (actually zip)
def yield_pairs(insts0, insts1):
    insts0, insts1 = list(insts0), list(insts1)
    if len(insts0) != len(insts1):
        zwarn(f"Input size mismatch: {len(insts0)} vs {len(insts1)}")
    yield from zip(insts0, insts1)

# check and iter sent pairs
def yield_sent_pairs(insts0, insts1):
    from .doc import Doc, Sent
    for inst0, inst1 in yield_pairs(insts0, insts1):
        if isinstance(inst0, Sent) and isinstance(inst1, Sent):
            yield (inst0, inst1)
        elif isinstance(inst0, Doc) and isinstance(inst1, Doc):
            if inst0.id != inst1.id:
                zwarn(f"DocID mismatch: {inst0.id} vs {inst1.id}")
            if len(inst0.sents) != len(inst1.sents):
                zwarn(f"Doc sent-num mismatch: {inst0.sents} vs {len(inst1.sents)}")
            for s0, s1 in zip(inst0.sents, inst1.sents):
                yield (s0, s1)
        else:
            raise RuntimeError(f"Err: Different/UNK types to eval {type(inst0)} vs {type(inst1)}")

# --
# nltk's tokenizer (for English)
class ModifiedTreebankWordTokenizer:
    def __init__(self):
        from nltk.tokenize import TreebankWordTokenizer
        self.base = TreebankWordTokenizer()

    def span_tokenize(self, sentence):
        import re
        raw_tokens = self.base.tokenize(sentence)
        # convert
        if ('"' in sentence) or ("''" in sentence):
            # Find double quotes and converted quotes
            matched = [m.group() for m in re.finditer(r"``|'{2}|\"", sentence)]
            # Replace converted quotes back to double quotes
            tokens = [matched.pop(0) if tok in ['"', "``", "''"] else tok for tok in raw_tokens]
        else:
            tokens = raw_tokens
        # align_tokens
        point = 0
        offsets = []
        for token in tokens:
            try:
                start = sentence.index(token, point)
            except ValueError as e:
                # raise ValueError(f'substring "{token}" not found in "{sentence}"') from e
                zwarn(f"Tokenizer skip unfound token: {token} ||| {sentence[point:]}")
                continue  # note: simply skip this one!!
            point = start + len(token)
            offsets.append((start, point))
        return offsets

class NLTKTokenizer:
    def __init__(self, scheme='default', word=None, sent=None):
        from nltk.tokenize import PunktSentenceTokenizer, WhitespaceTokenizer, RegexpTokenizer
        # --
        if scheme == 'default':
            _word, _sent = 'treebank', 'punkt'
        elif scheme == 'simple':  # simple scheme!
            _word, _sent = 'whitespace', 'newline'
        if word is not None:
            _word = word
        if sent is not None:
            _sent = sent
        # --
        self.word_toker = {'treebank': ModifiedTreebankWordTokenizer(), 'whitespace': WhitespaceTokenizer()}[_word]
        self.sent_toker = {'punkt': PunktSentenceTokenizer(), 'newline': RegexpTokenizer(r"\s*\n\s*", gaps=True)}[_sent]
        # --

    def tokenize(self, text: str, split_sent=True, return_posi_info=False):
        # first split sent
        if split_sent:
            sent_spans = list(self.sent_toker.span_tokenize(text))
            sents = [text[a:b] for a,b in sent_spans]
        else:
            sent_spans = [(0, len(text))]
            sents = [text]
        # then split tokens
        char2posi = [None] * len(text)  # int -> (sid, tid)
        all_tokens = []
        all_token_spans = []
        for sid, sent in enumerate(sents):
            tok_spans = list(self.word_toker.span_tokenize(sent))
            _toks = [sent[a:b] for a,b in tok_spans]
            _spans = []
            for ii, (a, b) in enumerate(tok_spans):
                _offset = sent_spans[sid][0]
                _s0, _s1 = _offset+a, _offset+b
                char2posi[_s0:_s1] = [(sid, ii)] * (b - a)
                _spans.append((_s0, _s1))
            all_tokens.append(_toks)
            all_token_spans.append(_spans)
        # Sent-> List[Tok], List[(start, end)], List[tid]
        # Doc=-> List[List[Tok]], List[List[(start, end)]], List[(sid, tid)]
        if not split_sent:
            all_tokens, all_token_spans, char2posi = \
                all_tokens[0], all_token_spans[0], [None if z is None else z[1] for z in char2posi]
        if return_posi_info:
            return all_tokens, all_token_spans, char2posi
        else:
            return all_tokens
        # --

# --
# label's getter/setter
def get_label_gs(trg_f: str):
    _trg_f = trg_f.strip()
    if _trg_f.startswith("_info:"):
        _field = _trg_f.split(":", 1)[1]
        _getter = lambda x: x.info[_field]
        _setter = lambda x, lab: x.info.update([(_field, lab)])
    elif _trg_f.startswith("lambda"):
        _f0, _f1 = (_trg_f.split("///", 1) + [None])[:2]  # note: special split!
        _getter = eval(_f0)
        _setter = eval(_f1) if _f1 else None
    else:
        _getter = lambda x: getattr(x, _trg_f)
        _setter = lambda x, lab: setattr(x, _trg_f, lab)
    return _getter, _setter
