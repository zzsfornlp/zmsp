#

# prompting with gpt3
import os.path
import sys
import re
from collections import Counter, defaultdict
import numpy as np
from msp2.nn import BK
from msp2.data.inst import yield_sents, set_ee_heads, SimpleSpanExtender, Sent
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.utils import Conf, zlog, init_everything, default_json_serializer, zopen, AlgoHelper
from msp2.tasks.zmtl3.mod.extract.evt_arg.onto import Onto

# --
class MainConf(Conf):
    def __init__(self):
        self.mode = ''  # p1(inst2query)/p2(query2res)/p3(res2pred)
        self.onto = ""  # ??
        self.input_file = ""
        self.input_ans = ""
        self.output_file = ""
        self.verbose = False
        # --

# --
def match_strs(l0, l1):
    ii0, ii1 = 0, 0
    while True:
        if ii0>=len(l0) or ii1>=len(l1):
            break
        if l0[ii0] == l1[ii1]:
            ii0 += 1
            ii1 += 1
        elif l0[ii0] == "":
            ii0 += 1
        elif l1[ii1] == "":
            ii1 += 1
        else:
            break
    return ii0, ii1

# --
def main(*args):
    conf = MainConf()
    conf: MainConf = init_everything(conf, args)
    cc = Counter()
    # --
    if conf.mode == 'p1':  # inst2query
        _questions = {
            'what': "What is the <R> of the <T> event?",
            'who': "Who is the <R> in the <T> event?",
            'where': "Where does the <T> event take place?",
            'where2': f"Where is the <R> of the <T> event?",
        }
        onto = Onto.load_onto(conf.onto)
        reader = ReaderGetterConf().get_reader(input_path=conf.input_file)
        all_outputs = []
        for inst in reader:
            cc['inst'] += 1
            for sent in yield_sents(inst):
                cc['sent'] += 1
                for evt in sent.events:
                    _toks = sent.seq_word.vals.copy()
                    _widx, _wlen = evt.mention.get_span()
                    _toks[_widx] = "\"" + _toks[_widx]
                    _toks[_widx+_wlen-1] = _toks[_widx+_wlen-1] + "\""
                    _trig_str = " ".join(_toks[_widx:_widx+_wlen])
                    cc['evt'] += 1
                    ff = onto.find_frame(evt.label)
                    for rr, _ in ff.role_map.values():
                        cc['query'] += 1
                        _qw = rr.qwords[0] if rr.qwords else 'what'
                        _question = _questions[_qw].replace("<T>", _trig_str).replace("<R>", rr.np)
                        _query = " ".join(_toks) + "\n\n" + f"Q: {_question}" + "\n\n" + f"A: The {rr.np} of the {_trig_str} event is"
                        qq = {"id": (inst.id, sent.id, evt.id), "role": rr.name, "query": _query}
                        cc['query_atoks'] += len(qq['query'].split()) * 1.333  # approximate
                        all_outputs.append(qq)
                        if conf.verbose:
                            zlog(f"#-- Query=\n{_query}")
        if conf.output_file:
            default_json_serializer.save_iter(all_outputs, conf.output_file)
    elif conf.mode == 'p2':  # query2answer
        from tqdm import tqdm
        import openai
        import json
        from transformers import AutoTokenizer
        # --
        qqs = default_json_serializer.load_list(conf.input_file)
        _output_file = conf.output_file
        if os.path.exists(_output_file):
            zlog(f"File existing: {_output_file}")
            x = input("Overwrite?(y/n): ")
            if x.strip() != 'y':
                zlog("No overwrite!!")
                exit()
        # --
        with zopen("_TMPK") as fd:
            openai.api_key = fd.read().strip()
        # --
        tokenizer = AutoTokenizer.from_pretrained('gpt2')  # work for both gpt2/3
        pre_allowed_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("not specified. unknown.\n"))
        with zopen(_output_file, 'w') as fd:
            for qq in tqdm(qqs):
                cc['query'] += 1
                cc['query_atoks'] += len(qq['query'].split()) * 1.333  # approximate
                _one_ids = tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize(" " + qq['query'].split("\n")[0])) + pre_allowed_ids
                response = openai.Completion.create(
                    engine="text-davinci-002",
                    prompt=qq['query'],
                    temperature=0,
                    max_tokens=16,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    logprobs=5,
                    stop=["\n"],
                    logit_bias={str(z):100 for z in _one_ids},
                )
                qq['response'] = dict(response)
                fd.write(json.dumps(qq) + "\n")
        # --
    elif conf.mode == 'p3':
        import string
        _punct = set(string.punctuation)
        import stanza
        _parser = stanza.Pipeline(processors='tokenize,pos,lemma,depparse', use_gpu=False)
        # --
        qqs = {(tuple(z['id']), z['role']): z for z in default_json_serializer.load_list(conf.input_ans)}
        onto = Onto.load_onto(conf.onto)
        reader = ReaderGetterConf().get_reader(input_path=conf.input_file)
        all_insts = list(reader)
        _extender = SimpleSpanExtender.get_extender('ef')
        for inst in all_insts:
            cc['inst'] += 1
            for sent in yield_sents(inst):
                sent_toks = ["".join([c for c in z if c not in _punct]) for z in sent.seq_word.vals]
                sent_pos = sent.seq_upos.vals
                cc['sent'] += 1
                for evt in sent.events:
                    # --
                    evt.clear_args()  # clear args
                    # --
                    cc['evt'] += 1
                    ff = onto.find_frame(evt.label)
                    _id_evt = (inst.id, sent.id, evt.id)
                    _cand_ans = defaultdict(list)  # role -> list(results)
                    _evt_widxes = list(range(evt.mention.widx, evt.mention.wridx))
                    for rr, _ in ff.role_map.values():
                        cc['arg'] += 1
                        qq = qqs[(_id_evt, rr.name)]
                        ans_str0 = " ".join(qq["response"]["choices"][0]["text"].strip().split("\n")[0].split())
                        ans_str = ''.join([c for c in ans_str0 if c not in _punct])
                        # match to original text
                        matched_widxes = []
                        if ans_str=="" or any(ans_str.startswith(z) for z in ['unknown', 'unspecified', 'not known', 'not specified']):
                            cc['arg_uns'] += 1
                        else:
                            # create answer sentence
                            res = _parser(ans_str)
                            words = res.sentences[0].words
                            _ans_sent = Sent.create([w.text for w in words])
                            _ans_sent.build_uposes([w.upos for w in words])
                            _ans_sent.build_dep_tree([w.head for w in words], [w.deprel for w in words])
                            # find root
                            ans_widxes = list(_ans_sent.tree_dep.chs_lists[0])
                            # find root's conj
                            for z in list(ans_widxes):
                                for _ch in _ans_sent.tree_dep.chs_lists[z+1]:
                                    if _ans_sent.tree_dep.seq_label.vals[_ch].split(":")[0] == 'conj':
                                        ans_widxes.append(_ch)
                            # filter by pos
                            ans_widxes = [z for z in ans_widxes if _ans_sent.seq_upos.vals[z] in {'VERB','NOUN','PROPN','PRON'}]
                            # match
                            for ii in ans_widxes:
                                try:
                                    ii2 = sent_toks.index(_ans_sent.seq_word.vals[ii])
                                    if ii2 not in _evt_widxes:
                                        matched_widxes.append(ii2)
                                except:
                                    pass
                        cc[f'arg_M0={len(matched_widxes)>0}'] += 1
                        for p_widx in matched_widxes[:2]:
                            _cand_ans[p_widx].append(
                                [np.mean(qq["response"]["choices"][0]["logprobs"]["token_logprobs"]), rr.name])
                        if conf.verbose:
                            zlog(f"#--\nquery={qq['query']}")
                            zlog(f"ctx={sent_toks}")
                            zlog(f"ans={ans_str}")
                    # --
                    # decide it
                    for p_widx, vs in _cand_ans.items():
                        vs.sort(reverse=True)
                        new_ef = sent.make_entity_filler(p_widx, 1)
                        _extender.extend_mention(new_ef.mention)
                        evt.add_arg(new_ef, role=vs[0][1])
                        cc['arg_M1'] += 1
        if conf.output_file:
            with WriterGetterConf().get_writer(output_path=conf.output_file) as writer:
                writer.write_insts(all_insts)
    else:
        raise NotImplementedError(f"UNK mode of {conf.mode}")
    # --
    zlog(f"Finished running with {conf.mode} from {conf.input_file} to {conf.output_file}: {cc}")
    # --

if __name__ == '__main__':
    main(*sys.argv[1:])

# --
"""
# debug
python3 -m msp2.tasks.zmtl3.scripts.misc.query_gpt3 mode:p1 onto:ace input_file:debug.json output_file:query.debug.json verbose:1
python3 -m msp2.tasks.zmtl3.scripts.misc.query_gpt3 mode:p2 input_file:query.debug.json output_file:res.debug.json
python3 -m msp2.tasks.zmtl3.scripts.misc.query_gpt3 mode:p3 onto:ace input_file:debug.json input_ans:res.debug.json output_file:pred.debug.json verbose:1
# --
# p1
python3 -m msp2.tasks.zmtl3.scripts.misc.query_gpt3 mode:p1 onto:ace input_file:../../events/data/data21f/en.ace2.test.json output_file:query.ace2.test.json
python3 -m msp2.tasks.zmtl3.scripts.misc.query_gpt3 mode:p1 onto:ere input_file:../../events/data/data21f/en.ere2.test.json output_file:query.ere2.test.json
# Finished running with p1 from ../../events/data/data21f/en.ace2.test.json to query.ace2.test.json: Counter({'query_atoks': 96646.49899999992, 'query': 1477, 'sent': 676, 'evt': 424, 'inst': 40})
# Finished running with p1 from ../../events/data/data21f/en.ere2.test.json to query.ere2.test.json: Counter({'query_atoks': 116068.30899999995, 'query': 1934, 'sent': 1163, 'evt': 551, 'inst': 31})
# --
# p2
python3 -m msp2.tasks.zmtl3.scripts.misc.query_gpt3 mode:p2 input_file:query.ace2.test.json output_file:res.ace2.test.json
python3 -m msp2.tasks.zmtl3.scripts.misc.query_gpt3 mode:p2 input_file:query.ere2.test.json output_file:res.ere2.test.json
# --
# p3
python3 -m msp2.tasks.zmtl3.scripts.misc.query_gpt3 mode:p3 onto:ace input_file:../../events/data/data21f/en.ace2.test.json input_ans:res.ace2.test.json output_file:pred.ace2.test.json
python3 -mpdb -m msp2.cli.analyze frame gold:../../events/data/data21f/en.ace2.test.json preds:pred.ace2.test.json
# Finished running with p3 from ../../events/data/data21f/en.ace2.test.json to pred.ace2.test.json: Counter({'arg': 1477, 'arg_M0=True': 1121, 'arg_M1': 811, 'sent': 676, 'evt': 424, 'arg_M0=False': 356, 'arg_uns': 58, 'inst': 40})
# lab-arg: 236.0/811.0=0.2910; 236.0/689.0=0.3425; 0.3147
# ==
python3 -m msp2.tasks.zmtl3.scripts.misc.query_gpt3 mode:p3 onto:ere input_file:../../events/data/data21f/en.ere2.test.json input_ans:res.ere2.test.json output_file:pred.ere2.test.json
python3 -mpdb -m msp2.cli.analyze frame gold:../../events/data/data21f/en.ere2.test.json preds:pred.ere2.test.json
#Finished running with p3 from ../../events/data/data21f/en.ere2.test.json to pred.ere2.test.json: Counter({'arg': 1934, 'arg_M0=True': 1311, 'sent': 1163, 'arg_M1': 877, 'arg_M0=False': 623, 'evt': 551, 'arg_uns': 167, 'inst': 31})
# lab-arg: 220.0/877.0=0.2509; 220.0/822.0=0.2676; 0.2590
"""
