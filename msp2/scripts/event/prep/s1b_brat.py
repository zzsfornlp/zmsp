#

# step 1: extract from brat
# -- still raw information!

# --
try:
    from s1_extract import *
except:
    from .s1_extract import *
# --

# --
def read_brat_doc(afile: str, tfile: str, doc_id_from_file: str, dataset_name: str):
    assert os.path.basename(afile).startswith(doc_id_from_file)
    assert os.path.basename(tfile).startswith(doc_id_from_file)
    # read source text
    with zopen(tfile) as fd:
        source_str = fd.read()
    # todo(+N): to read other ones!
    # read anns
    doc = new_doc(doc_id_from_file, dataset_name, source_str, {})
    with zopen(afile) as fd:
        ann_lines = [line.rstrip() for line in fd]
    # mentions (T: text-bound annotation)
    mentions = {}
    evt_mentions = set()
    for line in ann_lines:
        fields = line.split("\t")
        tag = fields[0]
        if tag.startswith("T"):  # mention
            mtype, mposi = fields[1].split(" ", 1)
            # todo(+N): currently simply take first and last
            mstart, mend = int(mposi.split(" ")[0]), int(mposi.split(" ")[-1])
            assert tag not in mentions
            mentions[tag] = (mtype, mstart, mend)
    # then all events
    events = {}
    for line in ann_lines:
        fields = line.split("\t")
        tag = fields[0]
        if tag.startswith("E"):  # event
            items = fields[1].split(" ")
            event_type, event_trigger = items[0].rsplit(":", 1)
            evt_mentions.add(event_trigger)
            mtype, mstart, mend = mentions[event_trigger]
            assert mtype == event_type
            # new evt
            cur_event = new_mention(tag, new_posi(mstart, mend-mstart), event_type, {})
            assert tag not in events
            events[tag] = cur_event
            cur_event["args"] = []
            # fill args
            for one_arg in items[1:]:
                arg_role, arg_mention = one_arg.rsplit(":", 1)
                cur_event["args"].append({"aid": arg_mention, "role": arg_role})
    # not-event mentions as efs
    entities = {}
    for mid, (mtype, mstart, mend) in mentions.items():
        if mid not in evt_mentions:
            cur_ef = new_mention(mid, new_posi(mstart, mend-mstart), mtype, {})
            assert mid not in entities
            entities[mid] = cur_ef
    # attributes
    for line in ann_lines:
        fields = line.split("\t")
        tag = fields[0]
        if tag.startswith("A"):  # attributes
            items = fields[1].split(" ")
            if len(items) < 3:
                items.append(True)  # implied binary value
            attr_name, attr_mid, attr_value = items
            if attr_mid in events:
                events[attr_mid]['info'][attr_name] = attr_value
    # --
    doc["ef_mentions"].extend([entities[e] for e in sorted(entities.keys())])
    doc["evt_mentions"].extend([events[e] for e in sorted(events.keys())])
    return doc

# --
def main(brat_dir: str, output_file: str, dataset_name="UNK"):
    zlog(f"Read brat from {brat_dir}")
    docs = []
    for file in sorted(os.listdir(brat_dir)):
        cur_doc_id = get_fname_id(file, [], [".ann"])
        if cur_doc_id is not None:
            full_afile = os.path.join(brat_dir, file)
            full_tfile = os.path.join(brat_dir, cur_doc_id+".txt")
            assert os.path.exists(full_tfile)
            cur_doc = read_brat_doc(full_afile, full_tfile, cur_doc_id, dataset_name)
            docs.append(cur_doc)
    zlog(f"End reading brat from {brat_dir}: {len(docs)} docs.")
    if output_file:
        write_dataset(docs, output_file)
    return docs

# --
# python3 s1b_brat.py [DIR_IN] [FILE_OUT] [name?]
# python3 -m msp2.scripts.event.s1b_brat [DIR_IN] [FILE_OUT] [name?]
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

# --
"""
# to run it all in once (extract+tokenize+convert+parse)
_zgo () {
python3 -m msp2.scripts.event.prep.s1b_brat ${DIR_IN} _tmp.s1.json
python3 -m msp2.scripts.event.prep.s2_tokenize -i _tmp.s1.json -o _tmp.s2.json --code en.plain
python3 -m msp2.scripts.event.prep.s3_convert _tmp.s2.json _tmp.s3.json
python3 -m msp2.cli.annotate 'stanza' stanza_use_gpu:0 stanza_lang:en stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:4 input_path:_tmp.s3.json output_path:${FILE_OUT}
}
"""
