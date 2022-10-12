#

# retokenize things for zh

import sys
from collections import Counter
from msp2.data.inst import yield_sents
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.utils import zlog, zwarn

def main(input_file: str, output_file: str, sep=''):
    from stanza.server import CoreNLPClient
    # --
    cc = Counter()
    reader = ReaderGetterConf().get_reader(input_path=input_file)
    with CoreNLPClient(annotators=['tokenize', 'ssplit'], timeout=60000, memory='16G', properties='zh') as client:
        with WriterGetterConf().get_writer(output_path=output_file) as writer:
            for inst in reader:
                cc['doc'] += 1
                # first retokenize things!
                for sent in yield_sents(inst):
                    cc['sent'] += 1
                    # collect
                    old2char = []  # old widx to char idx
                    ss = []
                    cur_ii = 0
                    for wstr in sent.seq_word.vals:
                        old2char.append((cur_ii, cur_ii+len(wstr)))
                        ss.extend([wstr, sep])
                        cur_ii += len(wstr) + len(sep)
                    ss = ss[:-1]  # no need last sep!
                    sss = ''.join(ss)
                    # tokenize
                    res = client.annotate(sss)
                    char2new = [None] * len(sss)
                    new_tokens = []
                    for _s in res.sentence:
                        for _t in _s.token:
                            _text, _start_char, _end_char = _t.originalText, _t.beginChar, _t.endChar
                            # special fix for some strange text like emoji ...
                            if _end_char - _start_char > len(_text):
                                zwarn(f"Strange token: {_text} [{_start_char}, {_end_char})")
                                _end_char = _start_char + len(_text)
                            # --
                            assert _text == sss[_start_char:_end_char]
                            char2new[_start_char:_end_char] = [len(new_tokens)] * (_end_char-_start_char)
                            new_tokens.append(_text)
                    # map
                    old2new = []
                    for _start_char, _end_char in old2char:
                        _news = sorted(set([z for z in char2new[_start_char:_end_char] if z is not None]))
                        assert len(_news) > 0
                        old2new.append(_news)
                    # update
                    sent.build_words(new_tokens)
                    sent._cache['old2new'] = old2new
                    # breakpoint()
                # reset item idxes!
                for sent in yield_sents(inst):
                    old2new = sent._cache['old2new']
                    for items, etype in zip([sent.entity_fillers, sent.events], ['ef', 'evt']):
                        for item in items:
                            cc[etype] += 1
                            widx, wlen = item.mention.get_span()
                            new_widxes = sorted(set(sum([old2new[z] for z in range(widx, widx+wlen)], [])))
                            item.mention.set_span(new_widxes[0], len(new_widxes))
                # --
                writer.write_inst(inst)
    zlog(f"Finish retok_zh from {input_file} to {output_file}: {cc}")
    # --

# python3 -m msp2.tasks.zmtl3.scripts.data2.retok_zh
if __name__ == '__main__':
    main(*sys.argv[1:])
