#

# run things streamingly (also with demo mode)

import shlex
from collections import Counter
from msp2.utils import zlog, zwarn, init_everything, Timer, ZObject, wrap_color
from msp2.nn.l3 import Zmodel
from msp2.data.inst import Doc, Sent, yield_frames, MyPrettyPrinter
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from ..core import ZOverallConf, TaskCenter, DataCenter, RunCenter
from .conf import conf_getter_test
from ..core import ZDataset

class SZOverallConf(ZOverallConf):
    def __init__(self):
        super().__init__()
        # --
        self.stream_input = ""
        self.stream_output = ""
        self.stream_bsize = 1  # stream batch size
        self.stream_show = False  # stop and show results
        self.stream_report_interval = 1000
        # --

# demo mode
def yield_input_insts():
    from nltk.tokenize import sent_tokenize, word_tokenize
    # --
    while True:
        # get context: str
        context = input("Input context: << ").strip()
        all_toks = [word_tokenize(t) for t in sent_tokenize(context)]
        all_sents = [Sent.create(ts) for ts in all_toks]
        doc = Doc.create(all_sents)
        # get pred/query: type trigger
        while True:
            pq0 = input("Input pred: << ").strip()
            _type, _pred = pq0.split(maxsplit=1)
            evt = None
            for sid, stoks in enumerate(all_toks):
                _preds = _pred.split()
                for wid in range(len(stoks)):
                    if stoks[wid:wid+len(_preds)] == _preds:  # find it!
                        evt = all_sents[sid].make_event(wid, 1, type=_type)
                        zlog(f"Annotate pred: {evt}")
                        break
            if evt is None:
                zlog(f"Failed locate the pred: {_type} {_pred}")
            else:
                if _type == 'Q':  # special for QA
                    # gather multiple questions
                    questions = []
                    while True:
                        question = input("Input Question: << ").strip()
                        if len(question) == 0: break
                        questions.append(question)
                    # put into many evts
                    evt.info['question'] = questions[0]  # at least one!
                    for qq in questions[1:]:
                        evt2 = evt.sent.make_event(evt.mention.shead_widx, 1, type=evt.type)
                        evt2.info['question'] = qq
                break
        # --
        yield doc
    # --

# show results
def show_outputs(insts):
    for evt in yield_frames(insts):
        arg_scores = None
        if 'arg_scores' in evt.info:
            arg_scores = evt.info['arg_scores']
            del evt.info['arg_scores']
        res = MyPrettyPrinter.str_frame(evt)
        if arg_scores is not None:
            evt.info['arg_scores'] = arg_scores
        zlog(res)
        zlog(f"#--\nThis is for {evt}: {evt.info.get('question')}")
        if 'arg_scores' in evt.info:
            for k in sorted(evt.info['arg_scores'].keys()):
                vs = evt.info['arg_scores'][k]
                ss = []
                top5_score = sorted(vs.values(), reverse=True)[:5][-1]
                for tok in evt.sent.tokens:
                    v = vs.get(tok.get_indoc_id(True), float('-inf'))
                    if v>=0:
                        v_str = wrap_color(str(v), bcolor='red')
                    elif v>=top5_score:
                        v_str = wrap_color(str(v), bcolor='blue')
                    else:
                        v_str = (str(v))
                    ss.append(f"{tok.word}({v_str})")
                zlog(f"Arg={k}: " + " ".join(ss))
            zlog("#--\n")
    # --

# --
def main(args):
    conf: SZOverallConf = init_everything(SZOverallConf(), args, sbase_getter=conf_getter_test)
    # --
    # task
    t_center = TaskCenter(conf.tconf)
    # vocab
    _tcf = t_center.conf
    t_center.load_vocabs(_tcf.vocab_load_dir)
    # model
    model = Zmodel(conf.mconf)
    t_center.build_mods(model)
    model.finish_build()  # note: build sr before possible loading in testing!!
    r_center = RunCenter(conf.rconf, model, t_center, None)
    if conf.rconf.model_load_name != "":
        r_center.load(conf.rconf.model_load_name)
    else:
        zwarn("No model to load, Debugging mode??")
    # --
    # stream data!
    zlog(f"Start (streamed-)testing {conf.stream_input} to {conf.stream_output}")
    _in_stream = iter(ReaderGetterConf().get_reader(input_path=conf.stream_input)) if conf.stream_input else yield_input_insts()
    writer = WriterGetterConf().get_writer(output_path=conf.stream_output) if conf.stream_output else None
    cc = Counter()
    cur_insts = []
    d = ZDataset(conf.dconf.test0, 'UNK', 'test', _no_load=True)  # simply use test0!
    cur_rr = Counter()
    while True:
        # first read more
        if len(cur_insts) == 0:
            while len(cur_insts) < conf.stream_bsize and _in_stream is not None:
                try:
                    inst = next(_in_stream)
                    cur_insts.append(inst)
                except StopIteration:
                    _in_stream = None
        # process
        if len(cur_insts) == 0:
            break  # finished
        cc['inst'] += len(cur_insts)
        # build a dataset on the fly
        d.set_insts(cur_insts)  # simply set one at a time
        for ibatch in d.yield_batches(loop=False):
            one_res = model(ibatch, do_pred=True)
            # --
            cur_rr['int_inst'] += len(cur_insts)
            cur_rr['int_batch'] += 1
            cur_rr += Counter(one_res)
            if cur_rr['int_batch'] % conf.stream_report_interval == 0:
                zlog(f"Test inst: {cur_rr}", timed=True)
                cur_rr.clear()
            # --
        # write?
        if writer is not None:
            writer.write_insts(cur_insts)
        # show & debug
        if conf.stream_show:
            show_outputs(cur_insts)
            breakpoint()
        # --
        cur_insts.clear()
    # --
    if writer is not None:
        writer.close()
    zlog(f"Finish streaming with {cc}")
    # --

# python3 -m msp2.tasks.zmtl3.main.stream device:0 'conf_sbase:data:ace;task:arg' "model_load_name:??/zmodel.best.m###DMarg0.qmod.emb_frame,DMarg0.qmod.emb_role" arg0.arg_mode:tpl arg0.mix_evt_ind:0.5 stream_input:?? stream_output:?? arg0.pred_store_scores:1 arg0.extend_span:0 stream_input: stream_show:1
if __name__ == '__main__':
    import sys
    with Timer(info=f"Streaming", print_date=True) as et:
        main(sys.argv[1:])
    # --
