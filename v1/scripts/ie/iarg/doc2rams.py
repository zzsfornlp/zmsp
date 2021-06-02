#

# convert from doc back to rams format (for scoring)

import sys, json
from tasks.zie.common.data import DocInstance, MyDocReader

#
def doc2rams(doc, lang):
    ret = {"doc_key": doc.doc_id, "language_id": lang, "source_url": None, "split": "UNK",
           "sentences": [z.words.vals[1:] for z in doc.sents], "predictions": []}
    # sidwid -> overall-tid
    cur_tid = 0
    posi2tid = []
    for sent in ret["sentences"]:
        slen = len(sent)
        posi2tid.append([z+cur_tid for z in range(slen)])
        cur_tid += slen
    # =====
    def mention2tids(mention, sents):
        sid, wid, wlen = mention.hard_span.position(headed=False)
        # anchor_sent = sents[sid]
        # wid2 = wid+wlen-1
        # assert wid>=1 and wid<anchor_sent.length
        # assert wid2>=1 and wid2<anchor_sent.length
        wid = max(1, wid)  # keep in sent bound
        tid = posi2tid[sid][wid-1]  # todo(note): here, -1 to exclude ROOT
        return [tid, tid+wlen-1]
    # =====
    # get the args
    all_evts = doc.events
    sents = doc.sents
    # assert len(all_evts) <= 1  # todo(note): this mode is for special decoding mode
    for one_evt in all_evts:
        one_res = [mention2tids(one_evt.mention, sents)]
        for one_arg in one_evt.links:
            one_res.append(mention2tids(one_arg.ef.mention, sents) + [one_arg.role, 1.])
        ret["predictions"].append(one_res)
    return ret

#
def main(input_f, output_f, lang):
    # fin = sys.stdin if (input_f=="-" or input_f=="") else open(input_f)
    fout = sys.stdout if (output_f=="-" or output_f=="") else open(output_f, "w")
    #
    for doc in MyDocReader(input_f, False, False):
        rams_doc = doc2rams(doc, lang)
        fout.write(json.dumps(rams_doc) + "\n")
    #
    # fin.close()
    fout.close()

if __name__ == '__main__':
    main(*sys.argv[1:])

# PYTHONPATH=../src/ python3 doc2rams.py INPUT OUTPUT eng
