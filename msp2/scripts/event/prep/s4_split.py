#

# split dataset
import os
import json
import numpy as np

np.random.seed(12345)

# --
# _EN_ACE_DEV = {'CNN_CF_20030303.1900.02', 'CNN_IP_20030329.1600.00-2', 'CNN_IP_20030402.1600.00-1', 'CNN_IP_20030405.1600.01-1', 'CNN_IP_20030409.1600.02', 'marcellapr_20050228.2219', 'rec.games.chess.politics_20041217.2111', 'soc.org.nonprofit_20050218.1902', 'FLOPPINGACES_20050217.1237.014', 'AGGRESSIVEVOICEDAILY_20041116.1347', 'FLOPPINGACES_20041117.2002.024', 'FLOPPINGACES_20050203.1953.038', 'TTRACY_20050223.1049', 'CNNHL_ENG_20030304_142751.10', 'CNNHL_ENG_20030424_123502.25', 'CNNHL_ENG_20030513_220910.32', 'CNN_ENG_20030304_173120.16', 'CNN_ENG_20030328_150609.10', 'CNN_ENG_20030424_070008.15', 'CNN_ENG_20030512_170454.13', 'CNN_ENG_20030620_085840.7', 'AFP_ENG_20030305.0918', 'AFP_ENG_20030311.0491', 'AFP_ENG_20030314.0238', 'AFP_ENG_20030319.0879', 'AFP_ENG_20030320.0722', 'AFP_ENG_20030327.0022', 'AFP_ENG_20030327.0224'}
_EN_ACE_DEV = {'CNN_CF_20030303.1900.02', 'CNN_IP_20030329.1600.00-2', 'CNN_IP_20030402.1600.00-1', 'CNN_IP_20030405.1600.01-1', 'CNN_IP_20030409.1600.02', 'marcellapr_20050228.2219', 'rec.games.chess.politics_20041216.1047', 'rec.games.chess.politics_20041217.2111', 'soc.org.nonprofit_20050218.1902', 'FLOPPINGACES_20050217.1237.014', 'AGGRESSIVEVOICEDAILY_20041116.1347', 'FLOPPINGACES_20041117.2002.024', 'FLOPPINGACES_20050203.1953.038', 'TTRACY_20050223.1049', 'CNNHL_ENG_20030304_142751.10', 'CNNHL_ENG_20030424_123502.25', 'CNNHL_ENG_20030513_220910.32', 'CNN_ENG_20030304_173120.16', 'CNN_ENG_20030328_150609.10', 'CNN_ENG_20030424_070008.15', 'CNN_ENG_20030512_170454.13', 'CNN_ENG_20030620_085840.7', 'AFP_ENG_20030304.0250', 'AFP_ENG_20030305.0918', 'AFP_ENG_20030311.0491', 'AFP_ENG_20030314.0238', 'AFP_ENG_20030319.0879', 'AFP_ENG_20030320.0722', 'AFP_ENG_20030327.0022', 'AFP_ENG_20030327.0224'}

_EN_ACE_TEST = {'AFP_ENG_20030401.0476', 'AFP_ENG_20030413.0098', 'AFP_ENG_20030415.0734', 'AFP_ENG_20030417.0004', 'AFP_ENG_20030417.0307', 'AFP_ENG_20030417.0764', 'AFP_ENG_20030418.0556', 'AFP_ENG_20030425.0408', 'AFP_ENG_20030427.0118', 'AFP_ENG_20030428.0720', 'AFP_ENG_20030429.0007', 'AFP_ENG_20030430.0075', 'AFP_ENG_20030502.0614', 'AFP_ENG_20030504.0248', 'AFP_ENG_20030508.0118', 'AFP_ENG_20030508.0357', 'AFP_ENG_20030509.0345', 'AFP_ENG_20030514.0706', 'AFP_ENG_20030519.0049', 'AFP_ENG_20030519.0372', 'AFP_ENG_20030522.0878', 'AFP_ENG_20030527.0616', 'AFP_ENG_20030528.0561', 'AFP_ENG_20030530.0132', 'AFP_ENG_20030601.0262', 'AFP_ENG_20030607.0030', 'AFP_ENG_20030616.0715', 'AFP_ENG_20030617.0846', 'AFP_ENG_20030625.0057', 'AFP_ENG_20030630.0271', 'APW_ENG_20030304.0555', 'APW_ENG_20030306.0191', 'APW_ENG_20030308.0314', 'APW_ENG_20030310.0719', 'APW_ENG_20030311.0775', 'APW_ENG_20030318.0689', 'APW_ENG_20030319.0545', 'APW_ENG_20030322.0119', 'APW_ENG_20030324.0768', 'APW_ENG_20030325.0786'}

STRATEGIES = {
    "en.ace": {
        "dev": ((lambda doc: doc['_id'] in _EN_ACE_DEV), None),
        "test": ((lambda doc: doc['_id'] in _EN_ACE_TEST), None),
    },
    "*.ace": {
        "dev": (None, 30), "test": (None, 40),
    },
    "*.ere": {
        "dev": ((lambda doc: 'LDC2016E73' in doc['info']['dataset']), 30),
        "test": ((lambda doc: 'LDC2017E54' in doc['info']['dataset']), None),
    },
}
# --

# --
def main(input_file: str, output_prefix: str, code=''):
    if code == '':
        code = os.path.basename(input_file)  # guess from input file
    cl, dset = code.split(".")[:2]
    code = f"{cl}.{dset}"
    s_key = {'en.ace': 'en.ace', 'en.ere': '*.ere', 'zh.ace': '*.ace',
             'zh.ere': '*.ere', 'es.ere': '*.ere', 'ar.ace': '*.ace'}
    strategies = STRATEGIES[s_key[code]]
    # --
    with open(input_file) as fd:
        docs = [json.loads(line) for line in fd]
    # --
    orig_len = len(docs)
    all_sets = {}
    for kk in ["dev", "test", "train"]:
        filter_f, sample_num = strategies.get(kk, (None, None))
        if filter_f is None:
            filter_f = (lambda x: True)
        # first filter all and get ids
        filter_docs = [d for d in docs if filter_f(d)]
        # sample?
        assert sample_num is None or sample_num <= len(filter_docs)
        if sample_num is not None and sample_num < len(filter_docs):
            np.random.shuffle(filter_docs)
            filter_docs = filter_docs[:sample_num]  # simply shuffle and keep
        # put and remove
        all_sets[kk] = filter_docs
        rm_doc_ids = set(z['_id'] for z in filter_docs)
        docs = [d for d in docs if d['_id'] not in rm_doc_ids]
    # write
    assert len(docs) == 0 and orig_len == sum(len(z) for z in all_sets.values())
    for kk, kk_docs in all_sets.items():
        output_file = f"{output_prefix}.{kk}.json"
        with open(output_file, 'w', encoding='utf8') as fd2:
            for d in kk_docs:
                fd2.write(json.dumps(d, ensure_ascii=False) + "\n")
            print(f"Write to {output_file} ({code}): {len(kk_docs)}")
    # --

# --
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

# --
# python3 s4_split.py ??.json ??
