#

# info

_T_SENTI = "Review: {input}\nSentiment{_head}:{_break}{_trg0} {label}{_trg1}"
_T_CLF = "Input: {input}\nType{_head}:{_break}{_trg0} {label}{_trg1}"
_T_NLI = "Premise: {premise}\nHypothesis: {hypothesis}\nPrediction{_head}:{_break}{_trg0} {label}{_trg1}"
_T_Q = "Question: {input}\nType{_head}:{_break}{_trg0} {label}{_trg1}"

_T_SS = [
    "Review: The label is positive.\nSentiment: positive\n\n",
    "Review: The label is negative.\nSentiment: negative\n\n"
]

# --
_TEMPLATES = {
'raw': {
    't_instance': "{_head}{input}{_break}",
},
'sst2': {  # plain one for generation
    't_instance': _T_SENTI,
},
'sst2v2': {  # plain one for generation
    't_instance': ''.join([_T_SS[z] for z in [0,1,1,0]]) + "Review: {input}\nSentiment{_head}:{_break}{_trg0} {label}{_trg1}",
},
'sst2M': {  # for mlm
    't_instance': "Review: {input}\nSentiment: {_head} {MASK_TOKEN}{_break}",
},
'sst2M2': {  # one more for mlm
    't_instance': "{input} It was {_head} {MASK_TOKEN} .{_break}",
    'extra_info': {'choices': ['terrible', 'great']},
},
'sst2R': {  # reverse
    't_instance': "Sentiment: {label}\nReview:{_break}{_trg0} {input}{_trg1}",
},
# --
'subj': {'t_instance': _T_CLF},
'mpqa': {'t_instance': _T_SENTI},
'agnews': {'t_instance': _T_CLF},
'cb': {'t_instance': _T_NLI},
'cr': {'t_instance': _T_SENTI},
'dbpedia': {'t_instance': _T_CLF},
'mr': {'t_instance': _T_SENTI},
'rte': {'t_instance': _T_NLI},
'trec': {'t_instance': _T_Q},
# --
}
_INFOS = {
'_base': {
    'choice_key': 'label',
    'choices': None,
    'mapper': {
        'input': (lambda inst, info: inst['sentence']),
        'label': (lambda inst, info: info['choices'][int(inst['label'])]),
    },
},
'sst2': {'choices': ['negative', 'positive']},
'subj': {'choices': ['subjective', 'objective']},
'mpqa': {'choices': ['negative', 'positive']},
'agnews': {
    'choices': ['world', 'sports', 'business', 'technology'],
    'mapper': {
        'input': (lambda inst, info: inst['sentence']),
        'label': (lambda inst, info: info['choices'][int(inst['label'])-1]),  # note: -1!
    },
},
'cb': {
    'choices': ['false', 'true', 'neither'],
    'mapper': {
        'premise': (lambda inst, info: inst['premise']),
        'hypothesis': (lambda inst, info: inst['hypothesis']),
        'label': (lambda inst, info: {'contradiction': 'false', 'entailment': 'true', 'neutral': 'neither'}[inst['label']]),
    },
},
'cr': {'choices': ['negative', 'positive']},
'dbpedia': {
    'choices': ['company', 'school', 'artist', 'athlete', 'politics', 'transportation', 'building', 'nature', 'village', 'animal', 'plant', 'album', 'film', 'book'],
    'mapper': {
        'input': (lambda inst, info: inst['sentence']),
        'label': (lambda inst, info: info['choices'][int(inst['label'])-1]),  # note: -1!
    },
},
'mr': {'choices': ['negative', 'positive']},
'rte': {
    'choices': ['false', 'true'],
    'mapper': {
        'premise': (lambda inst, info: inst['sentence_1']),
        'hypothesis': (lambda inst, info: inst['sentence_2']),
        'label': (lambda inst, info: {'not_entailment': 'false', 'entailment': 'true'}[inst['label']]),
    },
},
'trec': {'choices': ['description', 'entity', 'expression', 'human', 'location', 'number']},
}
# --
