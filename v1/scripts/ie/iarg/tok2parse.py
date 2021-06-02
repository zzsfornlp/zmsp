#

# step 3: further annotation
# using stanfordnlp for lemma upos udparsing
# also assign head for a span if there are not\
# todo(+N): currently ignore mwt, just assuming pre-tokenized

import json, sys
from typing import List, Tuple
import stanfordnlp

def zlog(s):
    print(str(s), file=sys.stderr, flush=True)

def zwarn(s):
    zlog("!!"+str(s))
    # set_trace()

#
class SParser:
    NOUN_HEAD_SCORES = {"NOUN": 0, "PROPN": -1, "NUM": -2, "VERB": -3, "PRON": -4, "ADJ": -7, "ADV": -10, "SYM": -11, "X": -12}
    VERB_HEAD_SCORES = {"VERB": 1, "NOUN": 0, "PROPN": -1, "NUM": -2, "PRON": -4, "ADJ": -7, "ADV": -10, "SYM": -11, "X": -12}

    # get final posi format: from List[(sid, wid)] -> (sid, wid+1, wlen)
    def get_posi(self, posi: List) -> Tuple:
        if posi is None:
            return None
        else:
            assert len(set([z[0] for z in posi])) == 1  # same sid
            sid, wid = posi[0]
            assert all(z[1] == wid + i for i, z in enumerate(posi))
            # todo(note): here += 1 for ROOT offset
            return sid, wid+1, len(posi)  # sid, wid, length

    # get distance-to-root with the heads
    def get_ddist(self, aug_heads: List):
        heads = aug_heads
        ddist = [-1] * len(heads)
        ddist[0] = 0
        for m in range(len(heads)):
            path = []
            cur_idx = m
            while ddist[cur_idx] < 0:
                path.append(cur_idx)
                cur_idx = heads[cur_idx]
            up_dist = ddist[cur_idx]
            for i, idx in enumerate(reversed(path)):
                ddist[idx] = up_dist + i + 1
        return ddist

    # heuristically find head
    def find_head(self, posi2: Tuple, sentences: List, is_event: bool):
        sid, wid, wlen = posi2
        idx_start, idx_end = wid, wid+wlen
        assert wlen>0
        if wlen==1:  # only one word
            return wid
        # todo(note): here we aug for a root
        cur_heads = [0] + sentences[sid]["governor"]
        cur_ddists = self.get_ddist(cur_heads)
        cur_poses = [None] + sentences[sid]["upos"]
        # todo(note): rule 1: simply find the highest node (nearest to root and not punct)
        # first pass by ddist
        min_ddist = min(cur_ddists[z] for z in range(idx_start, idx_end))
        cand_idxes1 = [z for z in range(idx_start, idx_end) if cur_ddists[z]<=min_ddist]
        assert len(cand_idxes1) > 0
        if len(cand_idxes1) == 1:
            return cand_idxes1[0]
        # next pass by POS
        POS_SCORES_MAP = SParser.VERB_HEAD_SCORES if is_event else SParser.NOUN_HEAD_SCORES
        pos_scores = [POS_SCORES_MAP.get(cur_poses[z], -100) for z in cand_idxes1]
        max_pos_score = max(pos_scores)
        cand_idxes2 = [v for i,v in enumerate(cand_idxes1) if pos_scores[i]>=max_pos_score]
        assert len(cand_idxes2) > 0
        if len(cand_idxes2) == 1:
            return cand_idxes2[0]
        # todo(note): rule 2: if same head and same pos, use the rightmost one
        # todo(+N): fine only for English?
        cand_idxes = cand_idxes2
        cand_heads, cand_poses = [cur_heads[z] for z in cand_idxes], [cur_poses[z] for z in cand_idxes]
        if all(z==cand_heads[0] for z in cand_heads) and all(z==cand_poses[0] for z in cand_poses):
            return cand_idxes[-1]
        if all(z=="PROPN" for z in cand_poses):
            return cand_idxes1[-1]
        if all(z=="NUM" for z in cand_poses):
            return cand_idxes1[-1]
        # todo(note): rule final: simply the rightmost
        if 1:
            cur_words = [None] + sentences[sid]["text"]
            ranged_words = cur_words[idx_start:idx_end]
            ranged_ddists = cur_ddists[idx_start:idx_end]
            ranged_heads = cur_heads[idx_start:idx_end]
            ranged_poses = cur_poses[idx_start:idx_end]
            zwarn(f"Cannot heuristically set head (is_event={is_event}), use the last one: "
                  f"{ranged_words} {ranged_ddists} {ranged_heads} {ranged_poses}")
        return cand_idxes[-1]

    def __init__(self, lang="en", models_dir=""):
        # mostly using stanfordnlp
        self.lang = lang
        if len(models_dir)>0:
            self.parser = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', lang=lang,
                                               tokenize_pretokenized=True, models_dir=models_dir)
        else:
            self.parser = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', lang=lang,
                                               tokenize_pretokenized=True)

    def parse_sents(self, sents: List):
        # batched!
        BATCH_SIZE = 16
        all_sentences = []
        for i in range(0, len(sents), BATCH_SIZE):
            one_res = self.parser(sents[i:i+BATCH_SIZE])
            all_sentences.extend(one_res.sentences)
        return all_sentences

    def parse_doc(self, doc):
        sents = doc["sents"]
        words_batch = [one_sent["text"] for one_sent in sents]
        results = self.parse_sents(words_batch)
        assert len(results) == len(sents)
        for one_res, one_sent in zip(results, sents):
            assert all(len(x.words)==1 for x in one_res.tokens)  # todo(note): pre-tokenized!!
            words = one_res.words
            assert [w.text for w in words] == one_sent["text"]
            one_sent.update({"lemma": [w.lemma for w in words], "upos": [w.upos for w in words],
                             "governor": [w.governor for w in words],
                             "dependency_relation": [w.dependency_relation for w in words]})
        # assign posi2 for each span
        for k in ["fillers", "entity_mentions", "event_mentions"]:
            is_event = (k=="event_mentions")
            if doc[k] is not None:
                for v in doc[k]:
                    to_assign = v["trigger"] if is_event else v
                    # todo(WARN): here wid+1 which means that posi2 also +1, but here only for checking,
                    #  not really used, therefore does not matter
                    posi_span = self.get_posi(to_assign["posi"])
                    if posi_span is None:
                        v["posi2"] = None
                        continue
                    orig_head = v.get("head")
                    if orig_head is not None:
                        posi_span_head = self.get_posi(orig_head["posi"])
                        if posi_span_head is None:
                            posi_span_head = posi_span
                    else:
                        posi_span_head = posi_span
                    if not (posi_span[0]==posi_span_head[0] and posi_span_head[1]>=posi_span[1]
                            and posi_span_head[1]+posi_span_head[2] <= posi_span[1]+posi_span[2]):
                        zwarn(f"Head span is not inside full span, use full posi instead: {posi_span} vs. {posi_span_head}")
                        posi_span_head = posi_span
                    # find head for the span
                    head_wid = self.find_head(posi_span_head, sents, is_event)
                    v["posi2"] = posi_span + (head_wid, )
        return doc

    def parse_fio(self, fin, fout):
        for line in fin:
            doc = json.loads(line)
            new_doc = self.parse_doc(doc)
            fout.write(json.dumps(new_doc)+"\n")

def main(input_f, output_f, lang, models_dir=""):
    zlog(f"Parse from {input_f} to {output_f} with lang={lang}")
    p = SParser(lang=lang, models_dir=models_dir)
    fin = sys.stdin if (input_f=="-" or input_f=="") else open(input_f)
    fout = sys.stdout if (output_f=="-" or output_f=="") else open(output_f, "w")
    p.parse_fio(fin, fout)
    fin.close()
    fout.close()

#
if __name__ == '__main__':
    main(*sys.argv[1:])

# tok -> parse
# for f in ../outputs_tok/*.json; do echo "#======"; echo dealing with $f; python3 s3_tok2parse.py $f ../outputs_parse/`basename $f` $cl; done |& tee ../outputs_parse/log
