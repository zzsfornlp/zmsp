#

# from oneie to my format

from msp2.utils import zlog, zwarn, zopen, default_json_serializer, OtherHelper, Random
from msp2.data.inst import Doc, Sent, Frame, Mention, ArgLink
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from collections import Counter

def convert_oneid(data_yielder, stat):
    rets = []
    prev_docid = None
    cur_sid = 0
    for dd in data_yielder:
        doc_id, sent_id = dd.get('doc_id'), dd['sent_id']
        _s0, _s1 = sent_id.rsplit("-", 1)
        if doc_id is None:
            doc_id = _s0
        # --
        if doc_id != prev_docid:
            stat["doc"] += 1
            doc = Doc.create(id=doc_id)
            rets.append(doc)
            cur_sid = 0
            if int(_s1) != 0:
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
        sent = Sent.create(dd['tokens'])
        doc.add_sent(sent)
        # --
        if 'pred' in dd:  # read predictions
            entities = []
            events = []
            for e in dd['pred']['entities']:
                start, end, etype = e[:3]
                ef = sent.make_entity_filler(start, end - start, type=etype)
                entities.append(ef)
                stat['ef'] += 1
            for e in dd['pred']['triggers']:
                start, end, etype = e[:3]
                evt = sent.make_event(start, end - start, type=etype)
                events.append(evt)
                stat['evt'] += 1
            for r in dd['pred']['roles']:
                i0, i1, role = r[:3]
                events[i0].add_arg(entities[i1], role=role)
                stat['evt_arg'] += 1
        else:  # read gold
            ef_map = {}  # id -> Frame
            for e in dd['entity_mentions']:
                ef = sent.make_entity_filler(e['start'], e['end']-e['start'], type=e['entity_type'], id=e['id'])
                assert e['id'] not in ef_map
                ef_map[e['id']] = ef
                stat['ef'] += 1
            for e in dd['event_mentions']:
                evt = sent.make_event(e['trigger']['start'], e['trigger']['end']-e['trigger']['start'], type=e['event_type'], id=e['id'])
                stat['evt'] += 1
                for a in e['arguments']:
                    a_ef = ef_map[a['entity_id']]
                    evt.add_arg(a_ef, role=a['role'])
                    stat['evt_arg'] += 1
        # --
    # --
    return rets

# --
def main(input_file: str, output_file: str, do_shuffle='0'):
    do_shuffle = int(do_shuffle)
    # --
    all_docs = []
    stat = Counter()
    yielder = default_json_serializer.yield_iter(input_file)
    all_docs.extend(convert_oneid(yielder, stat))
    if output_file:
        if do_shuffle:
            _gen = Random.get_generator('shuf')
            _gen.shuffle(all_docs)
            zlog(f"Shuffle docs: {len(all_docs)}")
        with WriterGetterConf.direct_conf(output_path=output_file).get_writer() as writer:
            writer.write_insts(all_docs)
    # --
    stat_str = OtherHelper.printd_str(stat, try_div=True)
    zlog(f"Read {input_file}, Write {output_file}, {len(all_docs)}: stat=\n{stat_str}")
    # --

# python3 oneie2mine.py IN OUT
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
