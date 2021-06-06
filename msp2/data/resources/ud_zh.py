#

# some ud maps

# note: in onto, there are three more: INF, URL, X
UPOS2CTB = {
    # Open class words
    "ADJ": ["VA", "JJ", ],
    "ADV": ["AD", "MSP", ],
    "INTJ": ["IJ", ],
    "NOUN": ["NT", "NN", "M", "ETC", "ON", ],
    "PROPN": ["NR", ],
    "VERB": ["VE", "VV", "LB", "SB", ],
    # Closed class words
    "ADP": ["LC", "P", "CS", "BA", ],
    "AUX": ["VC", ],
    "CCONJ": ["CC", ],
    "DET": ["DT", "INF", ],
    "NUM": ["CD", "OD", ],
    "PART": ["DEC", "DEG", "DER", "DEV", "AS", "SP"],
    "PRON": ["PN", ],
    "SCONJ": [],
    # Other
    "PUNCT": ["PU", ],
    "SYM": [],
    "X": ["FW", "X", "URL", ],
}
CTB2UPOS = {}  # reverse
for _t_ud, _ts_ctb in UPOS2CTB.items():
    for z in _ts_ctb:
        CTB2UPOS[z] = _t_ud
# --
