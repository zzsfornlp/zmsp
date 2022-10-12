#

# post-processing and stat

from collections import OrderedDict, Counter, defaultdict
from msp2.utils import zlog, zwarn, zopen, default_json_serializer, OtherHelper, Random
from msp2.data.inst import Doc, Sent, Frame, Mention, ArgLink
from msp2.data.vocab import SeqSchemeHelperStr
from msp2.data.rw import ReaderGetterConf, WriterGetterConf

def stat_docs(docs):
    cc = Counter()
    type_counters = defaultdict(Counter)
    for doc in docs:
        cc['doc'] += 1
        for sent in doc.sents:
            cc['sent'] += 1
            cc['tok'] += len(sent)
            # --
            # looking at ef
            for ef in sent.entity_fillers:
                cc['ef'] += 1
                cc['ef_link'] += len(ef.as_args)
                type_counters['ef'][ef.type] += 1
                # --
                for role_class in ['src', 'fact', 'time']:
                    _links = [z for z in ef.as_args if z.role.startswith(role_class)]
                    cc[f'ef_link_{role_class}'] += len(_links)
                    cc[f'ef_link_{role_class}_C={min(5,len(_links))}'] += 1
            # looking at evt
            for evt in sent.events:
                cc['evt'] += 1
                cc['evt_link'] += len(evt.args)
                type_counters['evt'][evt.type] += 1
                for arg in evt.args:
                    type_counters['arg'][arg.role] += 1
                # --
                for role_class in ['src', 'fact', 'time']:
                    _links = [z for z in evt.args if z.role.startswith(role_class)]
                    cc[f'evt_link_{role_class}'] += len(_links)
                    cc[f'evt_link_{role_class}_C={min(5,len(_links))}'] += 1
                    # distance
                    for _ll in _links:
                        _dist = min(5, abs(_ll.arg.sent.sid - evt.sent.sid))
                        cc[f'evt_link_{role_class}_D={_dist}'] += 1
                # --
                _fact_author = evt.info.get("fact_author", "Uu")  # some are missed in annotation
                cc[f'evt_fact_author_C={evt.info.get("fact_author") is not None}'] += 1
                _fact_src = [z for z in evt.info.get("fact_src", "").split("|") if z]
                cc[f'evt_fact_src_C={min(5, len(_fact_src))}'] += 1
                cc[f'evt_fact_srcS_C={min(5, len(set(_fact_src)))}'] += 1  # different labels?
                cc[f'evt_fact_asS_C={min(5, len(set(_fact_src+[_fact_author])))}'] += 1  # different labels: mix author+src
    # --
    return cc, type_counters

def main(method: str, input_file: str, output_file: str):
    # read
    reader = ReaderGetterConf.direct_conf(input_path=input_file).get_reader()
    docs = list(reader)
    input_cc, input_tcc = stat_docs(docs)
    zlog(f"#--\nRead {len(docs)} from {input_file}:")
    OtherHelper.printd(input_cc)
    OtherHelper.printd(input_tcc)
    # --
    # modify
    methods = method.split(",")
    m_cc = Counter()
    # keep only sip-related evts and links
    if "only_sip" in methods:
        for doc in docs:
            for sent in doc.sents:
                for evt in list(sent.events):
                    if not evt.type.startswith("SIP"):
                        sent.delete_frame(evt, 'evt')  # delete this one
                        m_cc['del_evt'] += 1
                    else:
                        for arg in list(evt.args):
                            if arg.role != "src.sip":
                                arg.delete_self()
                                m_cc['del_arg'] += 1
    # link fact-link to sip (if possible)
    if "link_fact" in methods:
        for doc in docs:
            for sent in doc.sents:
                for evt in sent.events:
                    for arg in evt.args:
                        if arg.role.startswith('fact.'):
                            # try to find sip
                            sip_links = [a for a in arg.arg.as_args if a.role=='src.sip' and a.main.type.startswith("SIP")]
                            m_cc[f'arg_cand_SL={len(sip_links)}'] += 1
                            for slink in sip_links:
                                slink.main.add_arg(evt, f'L_{arg.role}')
    # delete dummy "SOURCE_GEN"
    if "del_gen" in methods:
        for doc in docs:
            for sent in doc.sents:
                for ef in list(sent.entity_fillers):
                    m_cc[f'genef_{ef.type}'] += 1
                    if ef.type == "SOURCE_GEN":
                        sent.delete_frame(ef, 'ef')  # delete this one
                        m_cc['gen_del'] += 1
    # --
    # specific for the actual runnings
    # only keep the src and time links
    if "sip_src_time" in methods:
        for doc in docs:
            for sent in doc.sents:
                for evt in sent.events:
                    for arg in list(evt.args):
                        arg.role = arg.role.split(".")[0]
                        m_cc[f'arg_{arg.role}'] += 1
                        assert arg.role in ['fact', 'src', 'time']
                        if arg.role == 'fact':
                            arg.delete_self()
    # map "evt.info['fact_mix']"
    if "map_fact_mix" in methods:
        # true-certain, true-uncertain, false-certain, false-uncertain, unknown
        _MAP = {
            'CT+': 'true-certain', 'PR+': 'true-uncertain', 'PS+': 'true-uncertain',
            'CT-': 'false-certain', 'PR-': 'false-uncertain', 'PS-': 'false-uncertain',
            'Uu': 'unknown', 'NA': 'unknown', 'other': 'unknown', 'CTu': 'unknown', 'PRu': 'unknown', 'PSu': 'unknown',
        }
        for doc in docs:
            for sent in doc.sents:
                for evt in sent.events:
                    evt.info['map_fact_mix'] = _MAP[evt.info['fact_mix'].split('.')[-1]]
                    m_cc[evt.info['map_fact_mix']] += 1
    # --
    zlog(f"Modify with {methods}: {m_cc}")
    # --
    # output
    output_cc, output_tcc = stat_docs(docs)
    zlog(f"#--\nWrite {len(docs)} to {output_file}:")
    OtherHelper.printd(output_cc)
    OtherHelper.printd(output_tcc)
    # --
    if output_file:
        with WriterGetterConf.direct_conf(output_path=output_file).get_writer() as writer:
            writer.write_insts(docs)
    # --

# python3 pp_fb.py METHOD IN-DIR OUT-FILE
# python3 -m msp2.tasks.zmtl3.scripts.factbank.pp_fb METHOD IN-FILE OUT-FILE
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
