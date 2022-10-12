#

# prep qa for zh/ar

from collections import Counter, defaultdict
from msp2.tools.annotate import AnnotatorStanzaConf, AnnotatorStanza
from msp2.data.inst import yield_sents, yield_frames, Sent, Doc, Frame, set_ee_heads
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.utils import zlog, zwarn, default_json_serializer, Conf, init_everything, OtherHelper
from nltk.corpus import stopwords

# --
# note: from event/s2_tokenize.py
class CorenlpTokenizer:
    def __init__(self, lang: str):
        self.lang = lang
        self.is_zh = (self.lang == "zh")
        # --
        from stanza.server import CoreNLPClient
        self.client = CoreNLPClient(annotators=['tokenize', 'ssplit'],
                                    timeout=60000, memory='16G', properties=lang, be_quite=True)  # be_quite=True

    def __del__(self):
        self.client.stop()

    def tokenize(self, input_str: str):
        sents = self._tokenize(input_str, 0)
        # --
        char2posi = [None] * len(input_str)  # int -> (sid, tid)
        all_tokens = []
        all_token_spans = []
        for sid, sent in enumerate(sents):
            tok_spans = [(z[0], z[0]+z[1]) for z in sent['positions']]
            _toks0 = [input_str[a:b] for a, b in tok_spans]
            _toks = sent['tokens']
            if _toks != _toks0:
                zwarn(f"Diff tokens??: {_toks} vs {_toks0}")
            _spans = []
            for ii, (a, b) in enumerate(tok_spans):
                char2posi[a:b] = [(sid, ii)] * (b - a)
                _spans.append((a, b))
            all_tokens.append(_toks)
            all_token_spans.append(_spans)
        return all_tokens, all_token_spans, char2posi

    def _tokenize(self, input_str: str, offset: int):
        res = self.client.annotate(input_str)
        rets = []
        cur_idx = 0
        for sent in res.sentence:
            tokens, positions = [], []
            for tok in sent.token:
                # _text, _start_char, _end_char = tok.originalText, tok.beginChar, tok.endChar
                # special fix for some strange text like emoji ...
                # if _end_char - _start_char > len(_text):
                #     zwarn(f"Strange token: {_text} [{_start_char}, {_end_char})")
                #     _end_char = _start_char + len(_text)
                # assert _text == input_str[_start_char:_end_char]
                # --
                _text = tok.originalText
                try:
                    _start_char = input_str.index(_text, cur_idx)
                except:  # if cannot find ...
                    zwarn(f"Failed to find ``{_text}'' within ``{input_str[cur_idx:cur_idx+10]}''")
                    _start_char = cur_idx
                _end_char = _start_char + len(_text)
                cur_idx = _end_char
                tokens.append(_text)
                positions.append((_start_char+offset, _end_char - _start_char))
            rets.append({"tokens": tokens, "positions": positions})
        return rets
# --

# from "./prep_qa.py"
def read_squad(file: str, lang: str, filter_by_lang: bool):
    cc = Counter()
    full_lang_name = {'en': 'english', 'ar': 'arabic', 'zh': 'chinese'}[lang]
    toker = CorenlpTokenizer(lang)
    data = default_json_serializer.from_file(file)['data']
    docs = []
    for article in data:
        cc['article'] += 1
        title = article['title']
        for para_id, para in enumerate(article['paragraphs']):
            # --
            if filter_by_lang:
                if not any(qa['id'].startswith(full_lang_name) for qa in para['qas']):
                    continue
            # --
            doc_id = f"{title}_{para_id}"
            context = para['context']
            # --
            # tokenize and prepare context
            _tokens, _token_spans, _char2posi = toker.tokenize(context)
            cc['sent'] += len(_tokens)
            sents = [Sent.create(_toks) for _toks in _tokens]
            doc = Doc.create(sents, id=doc_id)
            docs.append(doc)  # actually paragraph
            # --
            # prepare questions
            for qa in para['qas']:
                cc['qas'] += 1
                question = qa['question']
                if qa.get('is_impossible', False):
                    cc['qas_A0'] += 1
                    assert len(qa['answers']) == 0
                    # --
                    frame = sents[0].make_event(0, 1, type='Q')  # note: simply put sent0!
                    frame.info['question'] = question
                else:
                    assert len(qa['answers']) > 0
                    cc[f"qas_A{min(2, len(qa['answers']))}"] += 1
                    _lcc = Counter([z["answer_start"] for z in qa['answers']])
                    # take the (max vote, shortest length) one!
                    _ans = sorted([(-_lcc[z["answer_start"]], len(z['text']), ii, z) for ii,z in enumerate(qa['answers'])])[0][-1]
                    # locate the answer span
                    _start, _end = _ans['answer_start'], _ans['answer_start'] + len(_ans['text'])
                    _tok_idxes = []
                    for _c in range(_start, _end):
                        _posi = _char2posi[_c] if _c<len(_char2posi) else None
                        if _posi is None: continue
                        if len(_tok_idxes)>0 and _tok_idxes[-1][0] != _posi[0]:
                            zwarn(f"Maybe bad sentence splitting for: {_ans}")
                            cc['qas_missplit'] += 1
                            break  # bad sentence splitting!
                        if len(_tok_idxes)==0 or _posi != _tok_idxes[-1]:
                            _tok_idxes.append(_posi)
                    # get the span
                    _sid, _widx, _wlen = _tok_idxes[0][0], _tok_idxes[0][1], _tok_idxes[-1][1] - _tok_idxes[0][1] + 1
                    _text = " ".join(_tokens[_sid][_widx:_widx+_wlen])
                    if ''.join(_text.split()) != ''.join(_ans['text'].split()):
                        zwarn(f"Maybe bad answer parsing: {_ans} vs {_text}")
                        cc['qas_mismatch'] += 1
                    # --
                    frame = sents[_sid].make_event(0, 1, type='Q')
                    frame.info['question'] = question
                    answer = sents[_sid].make_entity_filler(_widx, _wlen)
                    frame.add_arg(answer, role='A')
    # --
    zlog(f"Read squad from {file}: {cc}")
    return docs

def refind_targets(inst_stream, annotator, lang: str):
    cc = Counter()
    full_lang_name = {'en': 'english', 'ar': 'arabic', 'zh': 'chinese'}[lang]
    toker = CorenlpTokenizer(lang)
    _stopword_set = set(stopwords.words(full_lang_name)) if full_lang_name != 'chinese' else set(["、","。","〈","〉","《","》","一","一个","一些","一何","一切","一则","一方面","一旦","一来","一样","一种","一般","一转眼","七","万一","三","上","上下","下","不","不仅","不但","不光","不单","不只","不外乎","不如","不妨","不尽","不尽然","不得","不怕","不惟","不成","不拘","不料","不是","不比","不然","不特","不独","不管","不至于","不若","不论","不过","不问","与","与其","与其说","与否","与此同时","且","且不说","且说","两者","个","个别","中","临","为","为了","为什么","为何","为止","为此","为着","乃","乃至","乃至于","么","之","之一","之所以","之类","乌乎","乎","乘","九","也","也好","也罢","了","二","二来","于","于是","于是乎","云云","云尔","五","些","亦","人","人们","人家","什","什么","什么样","今","介于","仍","仍旧","从","从此","从而","他","他人","他们","他们们","以","以上","以为","以便","以免","以及","以故","以期","以来","以至","以至于","以致","们","任","任何","任凭","会","似的","但","但凡","但是","何","何以","何况","何处","何时","余外","作为","你","你们","使","使得","例如","依","依据","依照","便于","俺","俺们","倘","倘使","倘或","倘然","倘若","借","借傥然","假使","假如","假若","做","像","儿","先不先","光","光是","全体","全部","八","六","兮","共","关于","关于具体地说","其","其一","其中","其二","其他","其余","其它","其次","具体地说","具体说来","兼之","内","再","再其次","再则","再有","再者","再者说","再说","冒","冲","况且","几","几时","凡","凡是","凭","凭借","出于","出来","分","分别","则","则甚","别","别人","别处","别是","别的","别管","别说","到","前后","前此","前者","加之","加以","区","即","即令","即使","即便","即如","即或","即若","却","去","又","又及","及","及其","及至","反之","反而","反过来","反过来说","受到","另","另一方面","另外","另悉","只","只当","只怕","只是","只有","只消","只要","只限","叫","叮咚","可","可以","可是","可见","各","各个","各位","各种","各自","同","同时","后","后者","向","向使","向着","吓","吗","否则","吧","吧哒","含","吱","呀","呃","呕","呗","呜","呜呼","呢","呵","呵呵","呸","呼哧","咋","和","咚","咦","咧","咱","咱们","咳","哇","哈","哈哈","哉","哎","哎呀","哎哟","哗","哟","哦","哩","哪","哪个","哪些","哪儿","哪天","哪年","哪怕","哪样","哪边","哪里","哼","哼唷","唉","唯有","啊","啐","啥","啦","啪达","啷当","喂","喏","喔唷","喽","嗡","嗡嗡","嗬","嗯","嗳","嘎","嘎登","嘘","嘛","嘻","嘿","嘿嘿","四","因","因为","因了","因此","因着","因而","固然","在","在下","在于","地","基于","处在","多","多么","多少","大","大家","她","她们","好","如","如上","如上所述","如下","如何","如其","如同","如是","如果","如此","如若","始而","孰料","孰知","宁","宁可","宁愿","宁肯","它","它们","对","对于","对待","对方","对比","将","小","尔","尔后","尔尔","尚且","就","就是","就是了","就是说","就算","就要","尽","尽管","尽管如此","岂但","己","已","已矣","巴","巴巴","年","并","并且","庶乎","庶几","开外","开始","归","归齐","当","当地","当然","当着","彼","彼时","彼此","往","待","很","得","得了","怎","怎么","怎么办","怎么样","怎奈","怎样","总之","总的来看","总的来说","总的说来","总而言之","恰恰相反","您","惟其","慢说","我","我们","或","或则","或是","或曰","或者","截至","所","所以","所在","所幸","所有","才","才能","打","打从","把","抑或","拿","按","按照","换句话说","换言之","据","据此","接着","故","故此","故而","旁人","无","无宁","无论","既","既往","既是","既然","日","时","时候","是","是以","是的","更","曾","替","替代","最","月","有","有些","有关","有及","有时","有的","望","朝","朝着","本","本人","本地","本着","本身","来","来着","来自","来说","极了","果然","果真","某","某个","某些","某某","根据","欤","正值","正如","正巧","正是","此","此地","此处","此外","此时","此次","此间","毋宁","每","每当","比","比及","比如","比方","没奈何","沿","沿着","漫说","点","焉","然则","然后","然而","照","照着","犹且","犹自","甚且","甚么","甚或","甚而","甚至","甚至于","用","用来","由","由于","由是","由此","由此可见","的","的确","的话","直到","相对而言","省得","看","眨眼","着","着呢","矣","矣乎","矣哉","离","秒","称","竟而","第","等","等到","等等","简言之","管","类如","紧接着","纵","纵令","纵使","纵然","经","经过","结果","给","继之","继后","继而","综上所述","罢了","者","而","而且","而况","而后","而外","而已","而是","而言","能","能否","腾","自","自个儿","自从","自各儿","自后","自家","自己","自打","自身","至","至于","至今","至若","致","般的","若","若夫","若是","若果","若非","莫不然","莫如","莫若","虽","虽则","虽然","虽说","被","要","要不","要不是","要不然","要么","要是","譬喻","譬如","让","许多","论","设使","设或","设若","诚如","诚然","该","说","说来","请","诸","诸位","诸如","谁","谁人","谁料","谁知","贼死","赖以","赶","起","起见","趁","趁着","越是","距","跟","较","较之","边","过","还","还是","还有","还要","这","这一来","这个","这么","这么些","这么样","这么点儿","这些","这会儿","这儿","这就是说","这时","这样","这次","这般","这边","这里","进而","连","连同","逐步","通过","遵循","遵照","那","那个","那么","那么些","那么样","那些","那会儿","那儿","那时","那样","那般","那边","那里","都","鄙人","鉴于","针对","阿","除","除了","除外","除开","除此之外","除非","随","随后","随时","随着","难道说","零","非","非但","非徒","非特","非独","靠","顺","顺着","首先","︿","！","＃","＄","％","＆","（","）","＊","＋","，","０","１","２","３","４","５","６","７","８","９","：","；","＜","＞","？","＠","［","］","｛","｜","｝","～","￥"])
    from string import punctuation
    _stopword_set.update(punctuation)
    # --
    for inst in inst_stream:
        cc['all_inst'] += 1
        # --
        # note: need to cache all evts here
        inst_orig_evts = []
        for sent in yield_sents(inst):
            cc['all_sent'] += 1
            for evt in list(sent.events):  # list it since we want to change!
                cc['qas'] += 1
                inst_orig_evts.append(evt)
        # --
        for evt in inst_orig_evts:
            sent = evt.sent
            # --
            _question = evt.info['question']
            _q_toks = sum(toker.tokenize(_question.strip())[0], [])  # question tokens
            _q_sent = Sent.create(_q_toks)
            annotator.annotate([_q_sent])  # parse it!
            # --
            # check answer and context
            answer_lemmas = set(
                sum([[t.lemma.lower() for t in a.mention.get_tokens() if t.lemma is not None] for a in evt.args], []))
            ranked_preds = defaultdict(list)  # lemma -> pred_token
            # we might want to check all sents (for example, the paragraph in squad)
            all_sents = sent.doc.sents if sent.doc is not None else [sent]
            _sid0 = all_sents.index(sent)
            for _sii, _s in enumerate(all_sents):
                _dist = abs(_sii - _sid0)  # still make it closer to the answer
                for _t in _s.tokens:
                    _lemma = _t.lemma.lower() if _t.lemma is not None else None
                    if _lemma is None or _lemma in _stopword_set or _lemma in answer_lemmas:
                        continue
                    _depth = _s.tree_dep.depths[_t.widx]  # we want higher nodes
                    ranked_preds[_lemma].append((_dist, _sii, _depth, _t.widx, _t))  # add more tie-breakers
            for vs in ranked_preds.values():
                vs.sort()
            # --
            # check question
            _cand_qtoks = [
                t for t in _q_sent.tokens if (t.lemma is not None and t.lemma.lower() not in _stopword_set
                                              and t.lemma.lower() not in answer_lemmas)]
            _q_toks = sorted(_cand_qtoks, key=lambda t: (_q_sent.tree_dep.depths[t.widx], t.widx))
            trg_token = None
            for qt in _q_toks:
                cts = ranked_preds[qt.lemma.lower()]
                if len(cts) > 0:
                    trg_token = cts[0][-1]
                    if trg_token is evt.mention.shead_token:  # same token
                        cc['qas_nochange'] += 1  # no need to change!
                    else:
                        cc['qas_changed'] += 1
                        # --
                        # modify inplace!
                        new_evt = trg_token.sent.make_event(trg_token.widx, 1, type='Q')  # new target
                        new_evt.info.update(evt.info)  # note: remember to add info!
                        for arg in evt.args:
                            new_evt.add_arg(arg.arg, role='A')  # add answer
                        sent.delete_frame(evt, 'evt')
                        # --
                    break
            if trg_token is None:  # fail to find one, simply no change
                cc['qas_notfound'] += 1
    # --
    zlog(f"Do refind_targets: {OtherHelper.printd_str(cc, sep=' | ')}")
# --

# --
class MainConf(Conf):
    def __init__(self):
        self.data_type = 'squad'
        self.input_file = ''
        self.output_file = ''
        # --
        # special mode
        self.filter_by_lang = False
        self.lang = 'UNK'
        self.ann = AnnotatorStanzaConf.direct_conf(
            stanza_lang='UNK', stanza_use_gpu=False, stanza_processors="tokenize,pos,lemma,depparse".split(","),
            stanza_input_mode="tokenized",)
        self.do_refind = False
        # --

def main(*args):
    conf: MainConf = init_everything(MainConf(), args)

    if conf.data_type == 'json':
        reader = ReaderGetterConf().get_reader(input_path=conf.input_file)
        insts = list(reader)
    elif conf.data_type == 'squad':
        insts = read_squad(conf.input_file, conf.lang, conf.filter_by_lang)
    else:
        raise NotImplementedError(f"UNK data_type: {conf.data_type}")
    if conf.do_refind:
        conf.ann.stanza_lang = conf.lang
        annotator = AnnotatorStanza(conf.ann)
        refind_targets(insts, annotator, conf.lang)
    if conf.output_file:
        with WriterGetterConf().get_writer(output_path=conf.output_file) as writer:
            writer.write_insts(insts)
    # --

# python3 -m msp2.tasks.zmtl3.scripts.misc.prep_qa lang:? input_file:?
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

"""
# first download the original data
mkdir -p qaX
cd qaX
wget https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-train.json
wget https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-dev.json
wget https://raw.githubusercontent.com/ymcui/cmrc2018/master/squad-style-data/cmrc2018_train.json
wget https://raw.githubusercontent.com/ymcui/cmrc2018/master/squad-style-data/cmrc2018_dev.json
cd ..
# then process
{
# read
for wset in train dev; do
python3 -m msp2.tasks.zmtl3.scripts.misc.prep_qa2 lang:zh filter_by_lang:0 input_file:qaX/cmrc2018_${wset}.json output_file:qaX/zh.${wset}.json
python3 -m msp2.tasks.zmtl3.scripts.misc.prep_qa2 lang:ar filter_by_lang:1 input_file:qaX/tydiqa-goldp-v1.1-${wset}.json output_file:qaX/ar.${wset}.json
done
# concat
cat qaX/zh.{train,dev}.json >qaX/zh.all.json
cat qaX/ar.{train,dev}.json >qaX/ar.all.json
wc qaX/*.json
# parse
python3 -m msp2.cli.annotate 'stanza' stanza_use_gpu:0 stanza_lang:zh stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:1 input_path:qaX/zh.all.json "output_path:qaX/zh.all.ud2.json"
python3 -m msp2.cli.annotate 'stanza' stanza_use_gpu:0 stanza_lang:ar stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:1 input_path:qaX/ar.all.json "output_path:qaX/ar.all.ud2.json"
# refind
python3 -m msp2.tasks.zmtl3.scripts.misc.prep_qa2 lang:zh data_type:json do_refind:1 input_file:qaX/zh.all.ud2.json output_file:qaX/zhR.zll.ud2.json
python3 -m msp2.tasks.zmtl3.scripts.misc.prep_qa2 lang:ar data_type:json do_refind:1 input_file:qaX/ar.all.ud2.json output_file:qaX/arR.zll.ud2.json
} |& tee qaX/_log
# =>
Read squad from qaX/cmrc2018_train.json: Counter({'sent': 27823, 'qas': 10142, 'qas_A1': 10142, 'article': 2403, 'qas_mismatch': 463, 'qas_missplit': 242})
Read squad from qaX/tydiqa-goldp-v1.1-train.json: Counter({'sent': 57103, 'article': 49881, 'qas': 14805, 'qas_A1': 14805, 'qas_mismatch': 788, 'qas_missplit': 281})
Read squad from qaX/cmrc2018_dev.json: Counter({'sent': 10047, 'qas': 3219, 'qas_A2': 3219, 'article': 848, 'qas_mismatch': 138, 'qas_missplit': 31})
Read squad from qaX/tydiqa-goldp-v1.1-dev.json: Counter({'article': 5077, 'sent': 3159, 'qas': 921, 'qas_A2': 699, 'qas_A1': 222, 'qas_mismatch': 27, 'qas_missplit': 7})
Do refind_targets: all_inst: 3251 | all_sent: 37870 | qas: 13361 | qas_changed: 10877 | qas_nochange: 1405 | qas_notfound: 1079
Do refind_targets: all_inst: 15726 | all_sent: 60262 | qas: 15726 | qas_changed: 12518 | qas_nochange: 2586 | qas_notfound: 622
"""
