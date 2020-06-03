#

# transform the rams data to tokenized format

import sys
import json
import re
from copy import deepcopy

# cannot revert back to doc, but just fake it (but with src_doc, this is possible)
def msent2doc(msent, dataset_name, lang, src_doc=None, read_predictions=False):
    orig_sents = msent["sentences"]
    orig_source = "\n\n".join([" ".join(z) for z in orig_sents])
    if src_doc is None:
        doc = {"doc_id": msent["doc_key"], "dataset": dataset_name, "source": orig_source, "notag_source": orig_source,
               "entities": None, "relations": None, "hoppers": None,
               "fillers": [], "entity_mentions": [], "relation_mentions": None, "event_mentions": [],
               "lang": lang}
        doc["sents"] = [{"text": words, "positions": None} for words in orig_sents]
    else:
        # borrow src's sent info
        assert src_doc["doc_id"] == msent["doc_key"]
        assert src_doc["source"] == orig_source
        doc = deepcopy(src_doc)
        for k in ["fillers", "entity_mentions", "relation_mentions", "event_mentions"]:
            doc[k] = []  # clear these fields to feed in
    # =====
    # token-idx to sid
    tid2sid = []
    sid2offsets = []
    for sid, words in enumerate(orig_sents):
        sid2offsets.append(len(tid2sid))
        tid2sid.extend([sid]*len(words))
    # =====
    def _tid2posi2(tid0, tid1):
        sid = tid2sid[tid0]
        assert sid == tid2sid[tid1]
        wstart = tid0 - sid2offsets[sid]
        wlen = tid1-tid0+1  # todo(note): +1 since original is []
        return (sid, wstart, wlen)
    # =====
    # get all entity mentions (& args)
    # todo(note): only args are annotated and no ef types!
    ef_maps = {}  # (sid, wid, wlen) -> {...}
    for idx, one_ent in enumerate(msent["ent_spans"]):
        posi2 = _tid2posi2(one_ent[0], one_ent[1])
        cur_posi = [(posi2[0], wid) for wid in range(posi2[1], posi2[1]+posi2[2])]
        cur_v = {"id": "entity_"+str(len(ef_maps)), "gid": None, "offset": None, "length": None, "type": "unk", "mtype": None}
        cur_v["posi"] = cur_posi
        ef_key = posi2
        if ef_key in ef_maps:
            continue  # todo(note): repeated?
        ef_maps[ef_key] = cur_v
        doc["entity_mentions"].append(cur_v)
    # =====
    # get the only event mention and its args
    assert len(msent["evt_triggers"]) == 1
    for eidx, one_evt in enumerate(msent["evt_triggers"]):
        evt_posi2 = _tid2posi2(one_evt[0], one_evt[1])
        assert len(one_evt[-1]) == 1
        evt_type = one_evt[-1][0][0]
        cur_evt_posi = [(evt_posi2[0], wid) for wid in range(evt_posi2[1], evt_posi2[1]+evt_posi2[2])]
        cur_v = {"id": "event_"+str(eidx), "gid": None, "type": evt_type, "em_arg": []}
        cur_v["trigger"] = {"offset": None, "length": None, "posi": cur_evt_posi}
        # add arguments
        if read_predictions:
            predictions = msent["predictions"]
            assert len(predictions) == len(msent["evt_triggers"])
            cur_preds = predictions[eidx]
            a_evt_posi2 = _tid2posi2(*cur_preds[0])
            assert a_evt_posi2 == evt_posi2
            for one_arg in cur_preds[1:]:
                a_ef_posi2 = _tid2posi2(*one_arg[:2])
                a_role = one_arg[2]
                # =====
                # for pred ones, we need to add ents
                if a_ef_posi2 not in ef_maps:  # add one
                    ef_maps[a_ef_posi2] = {"id": "entity_"+str(len(ef_maps)), "gid": None, "offset": None, "length": None, "type": "unk", "mtype": None, "posi": [(a_ef_posi2[0], wid) for wid in range(a_ef_posi2[1], a_ef_posi2[1]+a_ef_posi2[2])]}
                    doc["entity_mentions"].append(ef_maps[a_ef_posi2])
                # =====
                arg_v = {"role": re.split(r'\d+', a_role)[-1], "aid": ef_maps[a_ef_posi2]["id"]}
                cur_v["em_arg"].append(arg_v)
        else:
            for one_arg in msent["gold_evt_links"]:
                a_evt_posi2 = _tid2posi2(*one_arg[0])
                assert a_evt_posi2 == evt_posi2
                a_ef_posi2 = _tid2posi2(*one_arg[1])
                a_role = one_arg[-1]
                # for gold ones, ef must be included already!!
                arg_v = {"role": re.split(r'\d+', a_role)[-1], "aid": ef_maps[a_ef_posi2]["id"]}
                cur_v["em_arg"].append(arg_v)
        # add one event
        doc["event_mentions"].append(cur_v)
    return doc

#
def main(input_f, output_f, dataset_name, lang, src_doc_f="", read_predictions=0):
    fin = sys.stdin if (input_f=="-" or input_f=="") else open(input_f)
    fout = sys.stdout if (output_f=="-" or output_f=="") else open(output_f, "w")
    # =====
    if src_doc_f:
        with open(src_doc_f) as fd:
            src_docs = [json.loads(line) for line in fd]
    else:
        src_docs = None
    read_predictions = bool(int(read_predictions))
    # =====
    for lidx, line in enumerate(fin):
        msent = json.loads(line)
        doc = msent2doc(msent, dataset_name, lang, src_doc=(src_docs[lidx] if src_docs is not None else None),
                        read_predictions=read_predictions)
        fout.write(json.dumps(doc) + "\n")
    #
    fin.close()
    fout.close()

if __name__ == '__main__':
    main(*sys.argv[1:])

# python3 s12_rams2tok.py <file-in> <file-out> RAMS lang
# python3 s12_rams2tok.py <file-in> <file-out> RAMS lang <src-in> 1
