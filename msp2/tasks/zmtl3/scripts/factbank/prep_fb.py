#

# reading factbank data
import sys
import os
from collections import OrderedDict, Counter
from msp2.utils import zlog, zwarn, zopen, default_json_serializer, OtherHelper, Random
from msp2.data.inst import Doc, Sent, Frame, Mention, ArgLink
from msp2.data.vocab import SeqSchemeHelperStr
from msp2.data.rw import ReaderGetterConf, WriterGetterConf

# general table reading
def read_tml(file):
    num_field = None
    rets = []  # List[List[??]]
    with zopen(file) as fd:
        for line in fd:
            line = line.strip()
            if len(line) > 0:
                fields = line.split("|||")
                if num_field is None:
                    num_field = len(fields)
                else:
                    assert num_field == len(fields)
                processed_fields = []
                for _f in fields:
                    if len(_f)>=2 and _f[0]=='\'' and _f[-1]=='\'':
                        _f = eval(_f)  # a string
                    else:
                        _f = int(_f)  # int
                    processed_fields.append(_f)
                rets.append(processed_fields)
    # --
    return rets

# check str
def check_str_match(mention, ss: str):
    a = ' '.join([z.word for z in mention.get_tokens()])
    a = ''.join(a.split()).lower()
    b = ''.join(ss.split()).lower()
    return a == b

# read "tokens_tml.txt"
def read_tokens(file, cc):
    docs = OrderedDict()  # doc_id -> Doc
    evts = OrderedDict()  # (doc_id, e_id) -> Evt
    efs = OrderedDict()  # (doc_id, e_id) -> TimeEf
    seq_helper = SeqSchemeHelperStr("BIO")
    # --
    cur_doc_id, cur_sid = None, None
    cur_sents = []
    cur_lines = []
    for fields in read_tml(file) + [None]:
        # judge the ending of a sent
        if fields is None or fields[0] != cur_doc_id or fields[1] != cur_sid:
            if len(cur_lines) > 0:
                assert all(z[0]==cur_doc_id for z in cur_lines)  # check doc-id
                # --
                # add placeholder sents in-between
                while cur_lines[0][1] > len(cur_sents):
                    zwarn(f"Add placeholder sent: {cur_doc_id} {cur_lines[0][1]} vs {len(cur_sents)}")
                    cur_sents.append(Sent.create(["."]))  # simply add a dot ...
                # --
                assert all(z[1]==len(cur_sents) for z in cur_lines)   # check s-id
                assert [z[2] for z in cur_lines] == list(range(len(cur_lines)))  # check t-id
                tokens = [z[3] for z in cur_lines]
                sent = Sent.create(tokens)
                cur_sents.append(sent)
                cc['sent'] += 1
                cc['tok'] += len(tokens)
                # read events
                evt_bio_tags = [f'{_fs[-1]}-{_fs[-2]}' if _fs[4]=='EVENT' else 'O' for _fs in cur_lines]
                evt_spans = seq_helper.tags2spans(evt_bio_tags)
                for _wid, _wlen, _id in evt_spans:
                    evt = sent.make_event(_wid, _wlen, type='EVENT', id=_id)
                    assert (cur_doc_id, _id) not in evts
                    evts[(cur_doc_id, _id)] = evt
                    cc['evt'] += 1
                # read time
                time_bio_tags = [f'{_fs[-1]}-{_fs[-2]}' if _fs[4]=='TIMEX3' else 'O' for _fs in cur_lines]
                time_spans = seq_helper.tags2spans(time_bio_tags)
                for _wid, _wlen, _id in time_spans:
                    ef = sent.make_entity_filler(_wid, _wlen, type='TIMEX3', id=_id)
                    assert (cur_doc_id, _id) not in efs
                    efs[(cur_doc_id, _id)] = ef
                    cc['time'] += 1
                # --
                cur_sid = None
                cur_lines = []
        # judge the ending of a doc
        if fields is None or fields[0] != cur_doc_id:
            if len(cur_sents) > 0:
                assert cur_doc_id not in docs
                doc = Doc.create(cur_sents, id=cur_doc_id)
                docs[cur_doc_id] = doc
                cc['doc'] += 1
                # --
                cur_doc_id = None
                cur_sents = []
        # read one
        if fields is not None:
            cur_lines.append(fields)
            cur_doc_id = fields[0]
            cur_sid = fields[1]
    # --
    return docs, evts, efs

# todo(+N): currently ignore DUMMY&AUTHOR

# read "fb_source.txt"
def read_source(file: str, docs, evts, efs, cc):
    for fields in read_tml(file):
        doc_id, sid, _id, wid, _tok = fields
        sent = docs[doc_id].sents[sid]
        if wid >= 0:
            ef = sent.make_entity_filler(wid, 1, type='SOURCE', id=_id)
            cc['ef_ok'] += 1
        elif wid == -2:  # put at a dummy place!!
            ef = sent.make_entity_filler(0, 1, type='SOURCE_GEN', id=_id)
            cc['ef_gen'] += 1
        else:
            cc['ef_ignored'] += 1
            continue
        if wid >= 0 and sent.seq_word.vals[wid] != _tok:
            zwarn(f"Wrong annotation ignored: {fields}")
            continue
        assert (doc_id, _id) not in efs
        efs[(doc_id, _id)] = ef
        cc['src_stored'] += 1
    # --
    # note: special fix
    efs[('wsj_0568.tml', 's28')] = efs[('wsj_0568.tml', 's29')]
    # --

# read "fb_sipAndSource.txt"
def read_sip(file: str, docs, evts, efs, cc):
    for fields in read_tml(file):
        doc_id, _, evt_id, _, evt_tok, src_id, src_tok = fields
        # --
        # first locate event
        sip_evt = evts[(doc_id, evt_id)]
        assert sip_evt.type == "EVENT" and check_str_match(sip_evt.mention, evt_tok)
        # then locate src
        sip_src = efs.get((doc_id, src_id))
        assert sip_src is not None
        # --
        if src_tok == "GEN":
            sip_evt.set_label("SIP_G")  # SIP of GEN
            cc['sip_g'] += 1
        else:
            assert check_str_match(sip_src.mention, src_tok)
            sip_evt.set_label("SIP_S")  # SIP of SOURCE
        # add link
        sip_evt.add_arg(sip_src, "src.sip")  # source of sip
        cc['sip_s'] += 1
        cc['link_sip'] += 1
    # --

# read "fb_factValue.txt"
def read_fact(file: str, docs, evts, efs, cc):
    for fields in read_tml(file):
        doc_id, _, _, evt_id, _, s_chain, evt_tok, st_chain, fact_label = fields
        # --
        # first locate evt
        fact_evt = evts[(doc_id, evt_id)]
        if not check_str_match(fact_evt.mention, evt_tok):
            assert ' '.join([z.word for z in fact_evt.mention.get_tokens()]).lower().startswith(evt_tok.lower())
            zwarn(f"Unmatched evt str: {fact_evt.mention} vs {evt_tok}")
        # then locate src
        comp_src_id = s_chain.split("_")[0]  # note: simply take the direct one!
        comp_src_token = st_chain.split("_")[0]
        for src_id, src_token in zip(comp_src_id.split("="), comp_src_token.split("=")):
            fact_src = efs.get((doc_id, src_id))
            if fact_src is None:
                # store author's one
                if src_token == "AUTHOR":
                    if fact_evt.info.get('fact_author') is not None:
                        zwarn(f"Repeated fact_author: {fact_evt.info.get('fact_author')} vs {fact_label}")
                    fact_evt.info["fact_author"] = fact_label
                    cc['fact_author'] += 1
                else:
                    assert src_token == "DUMMY"
                    cc['fact_dummy'] += 1
            else:
                if src_token == "GEN":
                    cc['fact_gen'] += 1
                else:
                    cc['fact_src'] += 1
                if 'fact_src' in fact_evt.info:
                    fact_evt.info['fact_src'] = '|'.join(sorted(fact_evt.info['fact_src'].split("|") + [fact_label]))
                else:
                    fact_evt.info['fact_src'] = fact_label
                # --
                # link
                assert check_str_match(fact_src.mention, src_token) or src_token == 'GEN'
                # add link
                fact_evt.add_arg(fact_src, f"fact.{fact_label}")  # source of sip
                cc['link_fact'] += 1
    # --
    # mix facts
    for one_evt in evts.values():
        fact_mix = one_evt.info.get('fact_src', '').split("|")[0]
        if not fact_mix:
            fact_mix = one_evt.info.get('fact_author', 'Uu')
            cc['evt_mix_author'] += 1
        else:
            cc['evt_mix_src'] += 1
        one_evt.info['fact_mix'] = fact_mix
    # --

# read "tml_tlink.txt"
def read_tlink(file: str, docs, evts, efs, cc):
    for fields in read_tml(file):
        doc_id, _, evt_id1, evt_id2, _, _, t_id1, t_id2, time_label, _, etext1, etext2, _ = fields
        _pat = [int(z!='') for z in [evt_id1, evt_id2, t_id1, t_id2]]
        if _pat == [1,0,0,1] or _pat == [0,1,1,0]:
            if _pat == [1,0,0,1]:
                _eid, _tid, _etext, _ttext = evt_id1, t_id2, etext1, etext2
            else:
                _eid, _tid, _etext, _ttext = evt_id2, t_id1, etext2, etext1
            if (doc_id, _tid) not in efs:
                zwarn(f"Unfound time-ef: {fields}")
                continue
            evt, ef = evts[(doc_id, _eid)], efs[(doc_id, _tid)]
            if not check_str_match(evt.mention, _etext):
                zwarn(f"Unmatched time-evt str: {evt.mention} vs {_etext}")
            if not check_str_match(ef.mention, _ttext):
                zwarn(f"Unmatched time-ef str: {ef.mention} vs {_ttext}")
            evt.add_arg(ef, f"time.{time_label}")
            cc[f'tlink_time'] += 1
        else:
            assert _pat == [1,1,0,0] or _pat == [0,0,1,1]
    # ==

def main(fb_dir: str, output_file: str):
    cc = Counter()
    # read them
    docs, evts, efs = read_tokens(os.path.join(fb_dir, 'tokens_tml.txt'), cc)
    read_source(os.path.join(fb_dir, 'fb_source.txt'), docs, evts, efs, cc)
    read_sip(os.path.join(fb_dir, 'fb_sipAndSource.txt'), docs, evts, efs, cc)
    read_fact(os.path.join(fb_dir, 'fb_factValue.txt'), docs, evts, efs, cc)
    read_tlink(os.path.join(fb_dir, 'tml_tlink.txt'), docs, evts, efs, cc)
    zlog(f"Read from {fb_dir}, cc={cc}")
    OtherHelper.printd(cc, sep=' || ')
    # --
    # write
    with WriterGetterConf.direct_conf(output_path=output_file).get_writer() as writer:
        writer.write_insts(docs.values())
    # --

# python3 prep_fb.py IN-DIR OUT-FILE
# python3 -m msp2.tasks.zmtl3.scripts.factbank.prep_fb IN-DIR OUT-FILE
if __name__ == '__main__':
    main(*sys.argv[1:])

# --
# steps
"""
export PYTHONPATH=??
# prepare them all
python3 -m msp2.tasks.zmtl3.scripts.factbank.prep_fb factbank_v1/data/annotation/ fb0.json |& tee _log1
python3 -m msp2.cli.annotate 'stanza' stanza_use_gpu:0 stanza_lang:en stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:1 input_path:fb0.json output_path:fb0.ud2.json
# --
python3 -m msp2.tasks.zmtl3.scripts.factbank.pp_fb 'del_gen,sip_src_time,map_fact_mix' fb0.ud2.json fb.ud2.json |& tee _log2
python3 -m msp2.scripts.event.prep.sz_stat input_path:fb.ud2.json |& tee _log3
# python3 -m pdb -m msp2.cli.analyze frame frame_getter:evt gold:??
# shuf & split
shuf fb.ud2.json >fb.shuf.ud2.json
head -n 198 fb.shuf.ud2.json >fb.train.ud2.json
tail -n 10 fb.shuf.ud2.json >fb.dev.ud2.json
"""
