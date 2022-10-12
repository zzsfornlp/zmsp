#

# simply convert evt-related things to my format
# note: currently only do ef/evt/evt-arg

from msp2.utils import zlog, zwarn, zopen, default_json_serializer, OtherHelper
from msp2.data.inst import Doc, Sent, Frame, Mention, ArgLink
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from collections import Counter

# --
def convert_mine(data: dict, stat):
    # first create doc
    # doc = Doc.create(text=data['text'], id=data['id'])
    doc = Doc.create(id=data['id'])
    doc.info.update(data['info'])
    doc.info['dataset'] = data['dataset']
    stat["doc"] += 1
    # then add sents
    for ss in data['sents']:
        sent = Sent.create(ss['tokens'])
        doc.add_sent(sent)
        stat["sent"] += 1
    all_sents = doc.sents
    # then add ef
    ef_map = {}  # id -> Frame
    for kk in ["ef", "evt"]:
        is_ef = (kk == "ef")
        for ee in data[f'{kk}_mentions']:
            sid, widx, wlen = ee["posi"]["posi_token"]
            if is_ef:
                item = all_sents[sid].make_entity_filler(widx, wlen, type=ee['type'])
            else:
                item = all_sents[sid].make_event(widx, wlen, type=ee['type'])
            item.info.update(ee["info"])
            # span position
            posi_token_code = ee['posi'].get('posi_token_code', '')
            if posi_token_code:
                item.mention.info['posi_token_code'] = posi_token_code
            # head position
            head_posi = ee["posi"].get("head")
            if head_posi is not None:
                assert is_ef
                h_sid, h_widx, h_wlen = head_posi["posi_token"]
                if h_sid != sid or not (h_widx>=widx and h_widx+h_wlen<=widx+wlen):  # no adding if not inside!!
                    zwarn(f"Head not inside span: {data['sents'][h_sid]['tokens'][h_widx:h_widx+h_wlen]} "
                          f"not in {data['sents'][sid]['tokens'][widx:widx+wlen]}")
                    head_posi = None
                else:
                    item.mention.set_span(h_widx, h_wlen, hspan=True)
            # --
            if is_ef:
                assert ee['id'] not in ef_map
                ef_map[ee['id']] = item
            stat[f"{kk}"] += 1
            stat[f"{kk}_{posi_token_code}"] += 1
            stat[f"{kk}_zhead_{int(head_posi is not None)}"] += 1
            # --
            # args for evt
            if not is_ef:
                for aa in ee["args"]:
                    arg_ef = ef_map[aa['aid']]
                    item.add_arg(arg_ef, aa['role'])  # add arg
                    arg_posi_token_code = arg_ef.mention.info.get('posi_token_code', '')
                    stat[f"{kk}_arg"] += 1
                    stat[f"{kk}_arg_{arg_posi_token_code}"] += 1
            # --
    # --
    return doc

# --
def convert_dygiepp(data: dict, stat):
    # first create doc
    doc = Doc.create(id=data['doc_key'])
    doc.info['dataset'] = data['dataset']
    stat["doc"] += 1
    # first build doc index
    doc_token_idx = []  # token-idx -> (sid, wid)
    for sid, ss in enumerate(data['sentences']):
        doc_token_idx.extend([(sid, ii) for ii in range(len(ss))])
    # then add sents
    for sid, ss in enumerate(data['sentences']):
        sent = Sent.create(ss)
        doc.add_sent(sent)
        stat["sent"] += 1
        # --
        # add ef
        ef_map = {}  # (start,end) -> Frame
        for idx0, idx1, tt in data['ner'][sid]:
            _sid, _widx = doc_token_idx[idx0]
            assert _sid == sid
            # assert (idx0,idx1) not in ef_map
            if (idx0, idx1) in ef_map:
                zwarn(f"Repeated ef in {data['ner'][sid]}")
                continue
            ef = sent.make_entity_filler(_widx, idx1-idx0+1, type=tt)
            ef_map[(idx0,idx1)] = ef
            stat['ef'] += 1
        # --
        # add evt
        for d_evt in data['events'][sid]:
            idx0, tt = d_evt[0]
            _sid, _widx = doc_token_idx[idx0]
            assert _sid == sid
            evt = sent.make_event(_widx, 1, type=tt)
            stat['evt'] += 1
            # args
            for a0, a1, rr in d_evt[1:]:
                arg_ef = ef_map[(a0, a1)]
                evt.add_arg(arg_ef, rr)  # add arg
                stat["evt_arg"] += 1
        # --
    # --
    return doc

# --
def convert_maven(data: dict, stat):
    # first create doc
    doc = Doc.create(id=data['id'])
    doc.info['title'] = data['title']
    stat["doc"] += 1
    # then add sents
    for sid, ss in enumerate(data['content']):
        sent = Sent.create(ss['tokens'])
        sent.build_text(ss['sentence'])  # also put original sentence
        doc.add_sent(sent)
        stat["sent"] += 1
        # --
    # --
    def _add_evt_mention(_mm: dict, _tt: str):
        sid = _mm['sent_id']
        widx0, widx1 = _mm['offset']
        assert doc.sents[sid].seq_word.vals[widx0:widx1] == _mm['trigger_word'].split()
        evt = doc.sents[sid].make_event(widx0, widx1-widx0, type=_tt)
        return evt
    # --
    # then add events
    for d_evt in data.get('events', []):
        for m in d_evt['mention']:
            _add_evt_mention(m, d_evt['type'])
            stat["evt_pos"] += 1
    # then add negative ones
    for d_neg in data.get('negative_triggers', []):
        _add_evt_mention(d_neg, 'NEG')
        stat["evt_neg"] += 1
    # then add candidate ones
    for d_cand in data.get('candidates', []):
        ee = _add_evt_mention(d_cand, 'TODO')
        ee.info['cand_id'] = d_cand['id']
        stat["evt_todo"] += 1
    # --
    return doc

# --
def main(input_file: str, output_file: str, convert_f="convert_mine"):
    # --
    convert_f = globals()[convert_f]
    # --
    all_docs = []
    stat = Counter()
    for doc in default_json_serializer.yield_iter(input_file):
        all_docs.append(convert_f(doc, stat))
    if output_file:
        with WriterGetterConf.direct_conf(output_path=output_file).get_writer() as writer:
            writer.write_insts(all_docs)
    # --
    stat_str = OtherHelper.printd_str(stat, try_div=True)
    zlog(f"Read {input_file}, Write {output_file}, {len(all_docs)}: stat=\n{stat_str}")
    # --

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

# --
# PYTHONPATH=../src/ python3 s3_convert.py <IN> <OUT> <??>
