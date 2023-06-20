#

# templates for generation

_TEMPLATES = {
'alpaca': {
    't_instance': [
        ((lambda x: bool(x['input'])), "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{_break}{_trg0}{output}"),
        (None, "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{_break}{_trg0}{output}"),
    ],
},
'flan': {
    't_instance': [
        ((lambda x: bool(x['input'])), "{instruction}\n\n{input}\n\n{_break}{_trg0}{output}"),
        (None, "{instruction}\n\n{_break}{_trg0}{output}"),
    ],
},
}

_INFOS = {
'_base': {
    'choice_key': "output",
    'choices': None,
    'mapper': {
        'instruction': (lambda inst, info: inst['instruction']),
        'input': (lambda inst, info: inst['input']),
        'output': (lambda inst, info: inst['output']),
    },
},
'alpaca': {},
'flan': {},
# --
# instruction for the clf tasks
'i_sst2': {
    'choices': ['negative', 'positive'],
    'mapper': {
        'instruction': (lambda inst, info: "Predict the sentiment of the following sentence, answering with positive or negative."),
        'input': (lambda inst, info: inst['sentence']),
        'output': (lambda inst, info: info['choices'][int(inst['label'])]),
    },
},
'i_subj': {
    'choices': ['subjective', 'objective'],
    'mapper': {
        'instruction': (lambda inst, info: "Predict the sentiment polarity of the following sentence, answering with subjective or objective."),
        'input': (lambda inst, info: inst['sentence']),
        'output': (lambda inst, info: info['choices'][int(inst['label'])]),
    },
},
'i_mpqa': {
    'choices': ['negative', 'positive'],
    'mapper': {
        'instruction': (lambda inst, info: "Predict the sentiment of the following sentence, answering with positive or negative."),
        'input': (lambda inst, info: inst['sentence']),
        'output': (lambda inst, info: info['choices'][int(inst['label'])]),
    },
},
'i_agnews': {
    'choices': ['world', 'sports', 'business', 'technology'],
    'mapper': {
        'instruction': (lambda inst, info: "Classify the news article into the following categories: world, sports, business or technology."),
        'input': (lambda inst, info: inst['sentence']),
        'output': (lambda inst, info: info['choices'][int(inst['label'])-1]),
    },
},
'i_cb': {
    'choices': ['false', 'true', 'neither'],
    'mapper': {
        'instruction': (lambda inst, info: "Judge whether Sentence 2 entails Sentence 1, answering with false (contradict), true (entail) or neither (neutral)."),
        'input': (lambda inst, info: f"Sentence 1: {inst['premise']} Sentence 2: {inst['hypothesis']}"),
        'output': (lambda inst, info: {'contradiction': 'false', 'entailment': 'true', 'neutral': 'neither'}[inst['label']]),
    },
},
'i_cr': {
    'choices': ['negative', 'positive'],
    'mapper': {
        'instruction': (lambda inst, info: "Predict the sentiment of the following sentence, answering with positive or negative."),
        'input': (lambda inst, info: inst['sentence']),
        'output': (lambda inst, info: info['choices'][int(inst['label'])]),
    },
},
'i_dbpedia': {
    'choices': ['company', 'school', 'artist', 'athlete', 'politics', 'transportation', 'building', 'nature', 'village', 'animal', 'plant', 'album', 'film', 'book'],
    'mapper': {
        'instruction': (lambda inst, info: "Classify the article into the following categories: company, school, artist, athlete, politics, transportation, building, nature, village, animal, plant, album, film or book."),
        'input': (lambda inst, info: inst['sentence']),
        'output': (lambda inst, info: info['choices'][int(inst['label'])-1]),
    },
},
'i_mr': {
    'choices': ['negative', 'positive'],
    'mapper': {
        'instruction': (lambda inst, info: "Predict the sentiment of the following sentence, answering with positive or negative."),
        'input': (lambda inst, info: inst['sentence']),
        'output': (lambda inst, info: info['choices'][int(inst['label'])]),
    },
},
'i_rte': {
    'choices': ['false', 'true'],
    'mapper': {
        'instruction': (lambda inst, info: "Judge whether Sentence 2 entails Sentence 1, answering with false (not_entailment) or true (entailment)."),
        'input': (lambda inst, info: f"Sentence 1: {inst['sentence_1']} Sentence 2: {inst['sentence_2']}"),
        'output': (lambda inst, info: {'not_entailment': 'false', 'entailment': 'true'}[inst['label']]),
    },
},
'i_trec': {
    'choices': ['description', 'entity', 'expression', 'human', 'location', 'number'],
    'mapper': {
        'instruction': (lambda inst, info: "Predict the category of the following question, answering with description, entity, expression, human, location or number."),
        'input': (lambda inst, info: inst['sentence']),
        'output': (lambda inst, info: info['choices'][int(inst['label'])]),
    },
},
}
