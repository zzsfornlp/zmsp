#

# from oneie to my format

from mspx.utils import zlog, zwarn, zopen, default_json_serializer, ZHelper, Random
from mspx.data.inst import Doc, Sent, Frame, Mention, ArgLink
from mspx.data.rw import ReaderGetterConf, WriterGetterConf
from collections import Counter

def convert_oneid(data_yielder, stat, aggr_doc: bool):
    rets = []
    prev_docid = None
    cur_sid = 0
    for dd in data_yielder:
        doc_id, sent_id = dd.get('doc_id'), dd.get('sent_id', dd.get("wnd_id", None))
        _s0, _s1 = sent_id.rsplit("-", 1) if '-' in sent_id else sent_id.rsplit("_", 2)[:2]
        if doc_id is None:
            doc_id = _s0
        # --
        if doc_id != prev_docid or (not aggr_doc):
            stat["doc"] += 1
            doc = Doc(id=doc_id) if aggr_doc else Doc()
            rets.append(doc)
            cur_sid = 0
            if aggr_doc and int(_s1) != 0:
                zwarn(f"Not continous {sent_id} vs {cur_sid}")
                cur_sid = int(_s1)
        else:
            assert _s0 == doc_id
            if int(_s1) != cur_sid:
                zwarn(f"Not continous {sent_id} vs {cur_sid}")
                cur_sid = int(_s1)
            doc = rets[-1]
        cur_sid += 1
        prev_docid = doc_id
        # --
        stat["sent"] += 1
        sent = Sent(dd['tokens'])
        doc.add_sent(sent)
        ef_map = {}  # id -> Frame
        for e in dd['entity_mentions']:
            ef = sent.make_frame(e['start'], e['end']-e['start'], e['entity_type'], 'ef', id=e['id'])
            assert e['id'] not in ef_map
            ef_map[e['id']] = ef
            stat['ef'] += 1
        for e in dd['event_mentions']:
            evt = sent.make_frame(e['trigger']['start'], e['trigger']['end']-e['trigger']['start'], e['event_type'], 'evt', id=e['id'])
            stat['evt'] += 1
            for a in e['arguments']:
                a_ef = ef_map[a['entity_id']]
                evt.add_arg(a_ef, a['role'])
                stat['evt_arg'] += 1
        for r in dd['relation_mentions']:
            assert len(r['arguments']) == 2  # binary relation!
            rel_args = {z['role']: z['entity_id'] for z in r['arguments']}
            assert len(rel_args) == 2
            ef1, ef2 = ef_map[rel_args['Arg-1']], ef_map[rel_args['Arg-2']]
            stat['rel'] += 1
            alink = ef1.add_arg(ef2, r['relation_type'])  # note: simply from A1 -> A2
            alink.info["is_rel"] = 1  # note: special mark
        # --
    # --
    return rets

# --
def main(input_file: str, output_file: str, aggr_doc='1', do_shuffle='0'):
    do_shuffle = bool(int(do_shuffle))
    aggr_doc = bool(int(aggr_doc))
    # --
    all_docs = []
    stat = Counter()
    yielder = default_json_serializer.yield_iter(input_file)
    all_docs.extend(convert_oneid(yielder, stat, aggr_doc))
    if output_file:
        if do_shuffle:
            _gen = Random.get_generator('shuf')
            _gen.shuffle(all_docs)
            zlog(f"Shuffle docs: {len(all_docs)}")
        with WriterGetterConf.direct_conf(output_path=output_file).get_writer() as writer:
            writer.write_insts(all_docs)
    # --
    stat_str = ZHelper.printd_str(stat, try_div=True)
    zlog(f"Read {input_file}, Write {output_file}, {len(all_docs)}: stat=\n{stat_str}")
    # --

# python3 -m mspx.scripts.data.evt.conv_oneie2mine IN OUT ...
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
