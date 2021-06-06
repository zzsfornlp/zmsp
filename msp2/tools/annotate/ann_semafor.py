#

# annotator with semafor

__all__ = [
    "AnnotatorSemaforConf", "AnnotatorSemafor", "SemaforHelper",
]

import os
from typing import List, Union, Dict
from msp2.data.inst import DataInstance, Doc, Sent, Mention, yield_sents
from msp2.utils import zlog, zwarn, zopen, system, default_json_serializer, StrHelper
from .base import AnnotatorConf, Annotator

# =====
"""
# prepare semafor
git clone https://github.com/Noahs-ARK/semafor
cd semafor; mvn package; cd ..
wget http://www.ark.cs.cmu.edu/SEMAFOR/semafor_malt_model_20121129.tar.gz
tar -zxvf semafor_malt_model_20121129.tar.gz
mkdir -p models
mv semafor_malt_model_20121129 models
# test
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-PUD/master/en_pud-ud-test.conllu
cat en_pud-ud-test.conllu | grep -o "text = .*" | cut -d "=" -f 2 >en_pud.txt
./semafor/bin/runSemafor.sh en_pud.txt "$PWD/en_pud.out" 4
"""

class AnnotatorSemaforConf(AnnotatorConf):
    def __init__(self):
        super().__init__()
        # specify semafor
        self.semafor_home = ""  # use "this/bin/runSemafor.sh" to run
        self.semafor_tmp_dir = "./"  # tmp dir (by default current dir)
        self.semafor_num_threads = 4
        self.semafor_log = ""  # output of semafor running
        # --
        self.semafor_use_cached = False  # simply skip step2 and use cached tmp files
        self.semafor_overwrite = False  # do not overwrite tmp files

@Annotator.reg_decorator("semafor", conf=AnnotatorSemaforConf)
class AnnotatorSemafor(Annotator):
    def __init__(self, conf: AnnotatorSemaforConf):
        super().__init__(conf)
        conf: AnnotatorSemaforConf = self.conf
        # --
        self.semafor_home = conf.semafor_home
        if self.semafor_home == "":
            # try to read env
            self.semafor_home = os.environ.get("SEMAFOR_HOME", "")
        assert self.semafor_home != "", "Please provide 'semafor_home': either by conf or ${SEMAFOR_HOME}"
        assert os.path.isfile(self.semafor_sh), "Cannot find semafor!"
        # --
        self.count = 0

    @property
    def semafor_sh(self):
        return os.path.join(self.conf.semafor_home, "bin", "runSemafor.sh")

    def delete_and_get_file(self, file: str, delete=True):
        conf: AnnotatorSemaforConf = self.conf
        if not conf.semafor_overwrite:
            file = f"{file}{self.count}"
        if delete and os.path.exists(file):
            os.remove(file)
        return file

    # note: should be run in large batch!
    def annotate(self, insts: List[DataInstance]):
        conf: AnnotatorSemaforConf = self.conf
        # --
        # get all sentences and run in batch
        all_sents = list(yield_sents(insts))
        # run all in batch
        # step 1: prepare input
        tmp_input = os.path.join(f"{conf.semafor_tmp_dir}", "_input.txt")
        tmp_input = os.path.abspath(tmp_input)  # require absolute path
        tmp_input = self.delete_and_get_file(tmp_input)
        with zopen(tmp_input, 'w') as fd:
            for sent in all_sents:  # write one line per sent
                fd.write(" ".join(sent.seq_word.vals)+"\n")
        # step 2: run semafor
        tmp_output = os.path.join(f"{conf.semafor_tmp_dir}", "_output.json")
        tmp_output = os.path.abspath(tmp_output)  # require absolute path
        tmp_output = self.delete_and_get_file(tmp_output, delete=(not conf.semafor_use_cached))
        if not conf.semafor_use_cached:  # otherwise simply skip running
            _semafor_log = conf.semafor_log if conf.semafor_log else "/dev/null"  # append to log!
            system(f"bash {self.semafor_sh} {tmp_input} {tmp_output} {conf.semafor_num_threads} >>{_semafor_log} 2>&1", ass=True)
        # step 3: read output and put them in sents
        semafor_results = default_json_serializer.load_list(tmp_output)
        assert len(semafor_results) == len(all_sents), "Error: predict inst number mismatch!"
        for one_res, one_sent in zip(semafor_results, all_sents):
            one_semafor_sent: Sent = SemaforHelper.semafor2sent(one_res)
            one_idx_map = SemaforHelper.find_sent_map(one_semafor_sent, one_sent)
            # put them back
            one_sent.clear_events()
            one_sent.clear_entity_fillers()
            # add them all
            for evt in one_semafor_sent.events:
                evt_widx, evt_wlen = evt.mention.widx, evt.mention.wlen
                mapped_posi = SemaforHelper.map_span(evt_widx, evt_wlen, one_idx_map)
                if mapped_posi is None:
                    zwarn(f"Failed mapping evt of {evt}: {evt.mention} to {one_sent.seq_word}")
                    continue
                evt2 = one_sent.make_event(mapped_posi[0], mapped_posi[1], type=evt.type)
                for alink in evt.args:
                    ef = alink.arg
                    ef_widx, ef_wlen = ef.mention.widx, ef.mention.wlen
                    mapped_posi = SemaforHelper.map_span(ef_widx, ef_wlen, one_idx_map)
                    if mapped_posi is None:
                        zwarn(f"Failed mapping arg of {alink}: {ef.mention} to {one_sent.seq_word}")
                        continue
                    ef2 = one_sent.make_entity_filler(mapped_posi[0], mapped_posi[1])  # make new ef for each arg
                    evt2.add_arg(ef2, role=alink.role)
        # --
        self.count += 1

# =====
# format helpers
class SemaforHelper:
    # one semafor dict to Sent object
    @staticmethod
    def semafor2sent(d: Dict):
        tokens = d["tokens"]
        ret = Sent.create(words=tokens)
        # -----
        def _read_mention(_spans):
            assert len(_spans)==1, "Assume single span!"
            _span = _spans[0]
            _start, _end, _text = _span["start"], _span["end"], _span["text"]
            assert StrHelper.delete_spaces(_text) == StrHelper.delete_spaces(''.join(tokens[_start:_end]))  # check without spaces
            return _start, _end-_start  # widx, wlen
        # -----
        for frame in d["frames"]:
            frame_target, frame_asets = frame["target"], frame["annotationSets"]
            # target
            evt_widx, evt_wlen = _read_mention(frame_target["spans"])
            evt = ret.make_event(evt_widx, evt_wlen, type=frame_target["name"])
            # roles
            assert len(frame_asets)==1 and frame_asets[0]["rank"]==0, "Assume only one rank=0 annotationSets!"
            for frame_role in frame_asets[0]["frameElements"]:
                ef_widx, ef_wlen = _read_mention(frame_role["spans"])
                ef = ret.make_entity_filler(ef_widx, ef_wlen)  # make new ef for each arg
                evt.add_arg(ef, role=frame_role["name"])
        return ret

    # find mappings: semafor_sent -> orig_sent
    @staticmethod
    def find_sent_map(semafor_sent: Sent, orig_sent: Sent):
        src_toks = semafor_sent.seq_word.vals
        trg_toks = orig_sent.seq_word.vals
        # --
        from msp2.tools.align import LCS
        # --
        def wpair_match_score_f(_a, _b, match_min_rate=0.25):  # matching prefix
            _c = sum(_x == _y for _x, _y in zip(_a, _b))
            _cr = sum(_x == _y for _x, _y in zip(reversed(_a), reversed(_b)))
            _s = (max(_c, _cr) / max(len(_a), len(_b)))  # will be 1. if perfectly match
            return _s if (_s >= match_min_rate) else 0.  # set matching thresh
        # --
        merge_to_a1, merge_to_a2, a1_to_merge, a2_to_merge = LCS.align_seqs(src_toks, trg_toks, match_f=wpair_match_score_f)
        ret_map = {}
        for i1, _ in enumerate(src_toks):
            i2 = merge_to_a2[a1_to_merge[i1]]  # matched target idx
            if i2 is not None:
                ret_map[i1] = i2
        return ret_map

    # map mention
    @staticmethod
    def map_span(widx: int, wlen: int, idx_map: Dict):
        target_idxes = []
        for i1 in range(widx, widx+wlen):
            i2 = idx_map.get(i1)
            if i2 is not None:
                target_idxes.append(i2)
        if len(target_idxes) == 0:
            return None
        min_v, max_v = min(target_idxes), max(target_idxes)
        return (min_v, max_v-min_v+1)

# example
#PYTHONPATH=../src/ python3 -m pdb -m msp2.cli.annotate 'semafor' input_path:_tmp.json output_path:_tmp2.json ann_batch_size:100 semafor_home:../semafor/semafor/ semafor_log:./_log semafor_use_cached:0
# pretty print
# cat _tmp.json | while read LINE; do echo $LINE | python3 -m json.tool; done
