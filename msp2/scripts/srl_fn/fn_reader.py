#

# read frames and corpus from fn's xml
# tested for both fn15 and fn17

from typing import List, Dict
import os
import sys
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape as xml_escape
from collections import Counter
from msp2.utils import zlog, zwarn, zopen, StrHelper, default_json_serializer, system
from msp2.data.inst import Doc, Sent, Frame
from msp2.data.rw import DataWriter, get_text_dumper

# =====
# helpers for reading

def assert_tag(node, name: str):
    assert node.tag == f"{{http://framenet.icsi.berkeley.edu}}{name}"

def ef_parse(file: str):
    with zopen(file) as fd:
        return ET.parse(fd)

class FnReadError(RuntimeError):
    def __init__(self, s):
        super().__init__(s)

# by default, preserve the sentence and no sentence tok!
_nltk_toker = None
def nltk_tokenize(s: str):
    global _nltk_toker
    if _nltk_toker is None:
        from nltk.tokenize.treebank import TreebankWordTokenizer
        _nltk_toker = TreebankWordTokenizer()
    tok_spans = _nltk_toker.span_tokenize(s)
    return [s[span[0]:span[1]] for span in tok_spans]

def get_subnodes(node, ch_name: str):
    return [n for n in node if n.tag.endswith(ch_name)]

def get_subnodes_by(node, f, only_one):
    rets = [n for n in node if f(n)]
    if only_one:
        assert len(rets) == 1
        return rets[0]
    else:
        return rets

def get_subnodes_map(node, f_key):
    ret = {}
    for n in node:
        k = f_key(n)
        if k in ret:
            ret[k].append(n)
        else:
            ret[k] = [n]
    return ret

# -----
# reading top level indexes

# frameIndex.xml
def read_frame_idxes(file: str):
    tree = ef_parse(file)
    node_frame_index = tree.getroot()
    assert_tag(node_frame_index, "frameIndex")
    rets = []
    for node_frame in node_frame_index:
        assert_tag(node_frame, "frame")
        fid, fname = node_frame.get("ID"), node_frame.get("name")
        fid = int(fid)
        rets.append({"ID": fid, "name": fname})
    zlog(f"Read frames from FrameIndex file {file}: {len(rets)}")
    return rets

# frRelation.xml
def read_frame_relations(file: str):
    tree = ef_parse(file)
    node_rel_index = tree.getroot()
    assert_tag(node_rel_index, "frameRelations")
    all_relations = {}  # rel_name -> List[{super: str, sub: str, roles: []}]
    for node_rels in node_rel_index:
        assert_tag(node_rels, "frameRelationType")
        rel_name = node_rels.get("name")
        rel_collections = []
        # --
        for node_rel in node_rels:
            assert_tag(node_rel, "frameRelation")
            rel_col = {"superFrameName": node_rel.get("superFrameName"), "subFrameName": node_rel.get("subFrameName"),
                       "FERelation": []}
            for node_role in node_rel:
                assert_tag(node_role, "FERelation")
                rel_col["FERelation"].append({"superFEName": node_role.get("superFEName"), "subFEName": node_role.get("subFEName")})
            rel_collections.append(rel_col)
        # --
        assert rel_name not in all_relations
        all_relations[rel_name] = rel_collections
    cc = Counter({k: len(v) for k, v in all_relations.items()})
    zlog(f"Read fr_rels from {file}: {cc}")
    return all_relations

# luIndex.xml
def read_lu_idxes(file: str):
    tree = ef_parse(file)
    node_lu_index = tree.getroot()
    assert_tag(node_lu_index, "luIndex")
    rets = []
    for node_lu in node_lu_index:
        if node_lu.tag.endswith("legend"):
            continue
        assert_tag(node_lu, "lu")
        r = {k: node_lu.get(k) for k in ["ID", "name", "frameName", "numAnnotInstances", "hasAnnotation"]}
        numAnnotInstances = 0 if r["numAnnotInstances"] is None else int(r["numAnnotInstances"])
        r.update({"ID": int(r["ID"]), "numAnnotInstances": numAnnotInstances, "hasAnnotation": (r["hasAnnotation"]=="true")})
        rets.append(r)
    # --
    cc = {"num_all": len(rets), "num_hasanno": len([lu for lu in rets if lu["hasAnnotation"]]),
          "num_anno": sum(lu["numAnnotInstances"] for lu in rets)}
    zlog(f"Read lu_idxes from LuIndex file {file}: {cc}")
    return rets

# semTypes.xml
def read_sem_types(file: str):
    tree = ef_parse(file)
    node_sems = tree.getroot()
    assert_tag(node_sems, "semTypes")
    rets = []
    for node_sem in node_sems:
        assert_tag(node_sem, "semType")
        r = {"ID": int(node_sem.get("ID")), "name": node_sem.get("name")}
        node_super_list = get_subnodes(node_sem, "superType")
        assert len(node_super_list) <= 1
        if len(node_super_list) == 0:
            r["superType"] = None
        else:
            r["superType"] = node_super_list[0].get("superTypeName")
        rets.append(r)
    # --
    cc = {"num_all": len(rets), "num_root": len([r for r in rets if r["superType"] is None])}
    zlog(f"Read sem_types from sem_types file {file}: {cc}")
    return rets

# fulltextIndex.xml
def read_fulltext_idxes(file: str):
    tree = ef_parse(file)
    node_ft = tree.getroot()
    assert_tag(node_ft, "fulltextIndex")
    rets = []
    for node_corpus in node_ft:
        assert_tag(node_corpus, "corpus")
        corpus_name = node_corpus.get("name")
        for node_doc in node_corpus:
            assert_tag(node_doc, "document")
            document_name = node_doc.get("name")
            # todo(note): in fn15, document_name is in "description"
            if document_name is None:
                document_name = node_doc.get("description")
            assert document_name is not None
            r = {"ID": int(node_doc.get("ID")), "name": f"{corpus_name}__{document_name}",
                 "description": node_doc.get("description")}
            rets.append(r)
    zlog(f"Read ft_idxes from fulltextIndex file {file}: {len(rets)}")
    return rets

# -----
# reading details

def read_frame(file: str):
    tree = ef_parse(file)
    node_frame = tree.getroot()
    assert_tag(node_frame, "frame")
    ret = {"ID": int(node_frame.get("ID")), "name": node_frame.get("name"), "FE": [], "lexUnit": []}
    # read others
    for node in node_frame:
        tag = node.tag
        if tag.endswith("FE"):
            assert_tag(node, "FE")
            node_sem_types = [z.get("name") for z in get_subnodes(node, "semType")]
            node_require_FEs = [z.get("name") for z in get_subnodes(node, "requiresFE")]
            node_exclude_FEs = [z.get("name") for z in get_subnodes(node, "excludesFE")]
            r = {"name": node.get("name"), "ID": int(node.get("ID")), "coreType": node.get("coreType"),
                 "semTypes": node_sem_types, "requiresFE": node_require_FEs, "excludesFE": node_exclude_FEs}
            ret["FE"].append(r)
        elif tag.endswith("lexUnit"):
            assert_tag(node, "lexUnit")
            basic = {"name": node.get("name"), "ID": int(node.get("ID"))}
            ret["lexUnit"].append(basic)
    return ret

def read_lu(file: str):
    tree = ef_parse(file)
    node_lu = tree.getroot()
    assert_tag(node_lu, "lexUnit")
    fname = node_lu.get("frame")
    ret = {"ID": int(node_lu.get("ID")), "name": node_lu.get("name"), "frame": fname}
    examplars = []
    # todo(note): ignore various other stuffs
    for node_sc in node_lu:
        if node_sc.tag.endswith("subCorpus"):
            for node_sent in node_sc:
                assert_tag(node_sent, "sentence")
                try:
                    sent = _build_sent(node_sent, fr_name=fname, lu_name=node_lu.get("name"))
                    examplars.append(sent)
                except FnReadError as e:
                    zwarn(f"Error when building sentence -> {e}")
    ret["exemplars"] = examplars
    return ret

# helpers
_SPECIAL_LAYER_SET = {"FE", "GF", "PT", "Sent", "Other", "Target"}  # these "layer" are treated specially, some are ignored
def _build_sent(node_sent, fr_name=None, lu_name=None):
    # get text and build toks
    node_text = get_subnodes(node_sent, "text")
    assert len(node_text) == 1
    # --
    read_text = node_text[0].text
    escape_text = xml_escape(read_text)
    if read_text != escape_text:
        zlog("Problem of xml escaping!")
    # todo(note): do we need to do xml escaping -> no_espace since less "Problem"
    text = read_text
    # text = escape_text
    # --
    # read annotations
    # ======
    # tokens
    node_unann = get_subnodes_by(node_sent, lambda n: any(n2.get("name") in ("BNC", "PENN") for n2 in n), True)
    unann_layer_maps = get_subnodes_map(node_unann, lambda n: n.get("name"))
    # POS layer
    node_tok_layer = unann_layer_maps.get("BNC", []) + unann_layer_maps.get("PENN", [])
    assert len(node_tok_layer) == 1
    node_tok_layer = node_tok_layer[0]
    toks, start_map, end_map = _build_toks(text, node_tok_layer)
    # WSL layer
    if "WSL" in unann_layer_maps:
        wsls = ["O"] * len(toks)
        assert len(unann_layer_maps["WSL"]) == 1
        node_wsl_layer = unann_layer_maps["WSL"][0]
        for node in node_wsl_layer:
            wsl_label = node.get("name")
            widx, wlen = _get_span(node, start_map, end_map, text, toks)
            for ii in range(widx, widx+wlen):
                # assert wsls[ii] == "O"
                wsls[ii] = wsl_label
    else:
        # no WSL in fn15.lu
        wsls = None
    # build sent
    ret_sent = Sent.create(toks)
    ret_sent.wsl = wsls  # directly assign this
    # =====
    # annos of frames
    # todo(+W): currently ignore cxn*
    list_node_ann = get_subnodes_by(node_sent, lambda n: any(n2.get("name")=="Target" for n2 in n) and n.get("cxnName") is None, False)
    for node_ann in list_node_ann:
        ann_layer_maps = get_subnodes_map(node_ann, lambda n: n.get("name"))
        fname, luname = node_ann.get("frameName", fr_name), node_ann.get("luName", lu_name)
        assert fname is not None and luname is not None
        # target
        all_posi_targets = []
        for target_layer in ann_layer_maps["Target"]:  # sometimes there are multiple targets
            for node_layer_target_label in get_subnodes_by(target_layer, lambda *args: True, False):
                assert node_layer_target_label.get("name") == "Target"
                posi_target = _get_span(node_layer_target_label, start_map, end_map, text, toks)
                all_posi_targets.append(posi_target)
        all_posi_targets = sorted(set(all_posi_targets))  # there can be repeated ones!
        # --
        if len(all_posi_targets) == 0:
            zwarn("Problem of targets: no targets, skip this frame!!")
            continue
        # --
        final_posi_target = [all_posi_targets[0][0], sum(all_posi_targets[-1])-all_posi_targets[0][0]]
        final_posi_target_valids = [0] * final_posi_target[-1]
        for one_posi_target in all_posi_targets:
            for ii in range(one_posi_target[0]-final_posi_target[0], sum(one_posi_target)-final_posi_target[0]):
                final_posi_target_valids[ii] = 1
        if not all(final_posi_target_valids):
            print_info = [f"{toks[ii+final_posi_target[0]]}" if final_posi_target_valids[ii] else f"[{toks[ii+final_posi_target[0]]}]"
                          for ii in range(len(final_posi_target_valids))]
            zwarn(f"Problem of discontinuous target span: {print_info}")
        # =====
        evt = ret_sent.make_event(final_posi_target[0], final_posi_target[1], type=fname)
        iargs = []  # List[{role, itype}]
        sargs = []  # List[{role, posi}]
        evt.info.update({"frameName": fname, "luName": luname, "iargs": iargs, "status": node_ann.get("status")})
        if not all(final_posi_target_valids):  # mark discontinous trigger
            evt.info.update({"DTrgs": [final_posi_target[0]+ii for ii,vv in enumerate(final_posi_target_valids) if vv]})
        # FE
        assert len(ann_layer_maps["FE"]) <= 3, "At most three layers of FE!!"
        for node_fe_rank, node_layer_fe in enumerate(ann_layer_maps["FE"], start=1):
            assert node_fe_rank == int(node_layer_fe.get("rank"))
            # TODO(+W): how to solve discontinous ones?
            for node_fe in node_layer_fe:
                role = node_fe.get("name")
                if "itype" in node_fe.attrib:
                    iargs.append({"role": role, "itype": node_fe.get("itype")})
                    # if role not in iargs or iargs[role] == node_fe.get("itype"):
                    #     iargs[role] = node_fe.get("itype")
                    # else:
                    #     zwarn(f"Problem of conflicted itype {role}: {iargs[role]} vs. {node_fe.get('itype')}")
                else:
                    posi_fe = _get_span(node_fe, start_map, end_map, text, toks)
                    # todo(note): make separate efs
                    ef = ret_sent.make_entity_filler(posi_fe[0], posi_fe[1], type=f"UNK")
                    arglink = evt.add_arg(ef, role=role)
                    arglink.info["rank"] = node_fe_rank
        # other FE like modifiers
        for other_name, other_anns in ann_layer_maps.items():
            if other_name not in _SPECIAL_LAYER_SET:
                # only one (rank1) such layer
                assert len(other_anns)==1, f"Meet multiple layers of {other_name}"
                other_ann_layer = other_anns[0]
                assert int(other_ann_layer.get("rank")) == 1
                for node_mod in other_ann_layer:
                    role = f"{other_name}.{node_mod.get('name')}"
                    posi_mod = _get_span(node_mod, start_map, end_map, text, toks)
                    sargs.append({"role": role, "posi": posi_mod})  # same sentence!
        # direct assign
        evt.iargs = iargs
        evt.sargs = sargs
        # todo(note): ignore other things!
    # =====
    return ret_sent

def _build_toks(text: str, node_tok):
    # todo(note): ignore POS
    # toks = text.split()
    toks = nltk_tokenize(text.strip())  # use NLTK since still not everything is tokenized!
    split_positions = StrHelper.index_splits(text, toks)
    # read positions from xml
    xml_positions = []
    for n in node_tok:
        assert_tag(n, "label")
        i0, i1 = int(n.get("start")), int(n.get("end"))
        xml_positions.append((i0, i1-i0+1))
    xml_positions.sort()
    if split_positions != xml_positions:
        # todo(note): simply ignore this tok layer!
        # zwarn("Problem in toks positions: use split ones!!")
        pass
    # check finished!
    start_map, end_map = {p[0]:i for i,p in enumerate(split_positions)}, {p[0]+p[1]:i for i,p in enumerate(split_positions)}
    return toks, start_map, end_map

def _get_span(node, start_map, end_map, text: str, toks: List[str]):
    # =====
    def _cidx2widx(_cidx: int, _map: Dict, _inc: int):
        _mismatch = False
        _orig_cidx = _cidx
        _widx = _map.get(_cidx)
        if _widx is None:
            _mismatch = True
            _upper = max(_map.keys())
            while _cidx>=0 and _cidx<=_upper:
                _cidx += _inc
                _widx = _map.get(_cidx)
                if _widx is not None:
                    break
            # zwarn(f"Exact char idx failed: {_orig_cidx} -> {_cidx}")
        if _widx is None:
            raise FnReadError("Index error!")
        return _widx, _mismatch
    # --
    assert_tag(node, "label")
    char_start, char_end = int(node.get("start")), int(node.get("end")) + 1
    tok_start, mis1 = _cidx2widx(char_start, start_map, -1)
    tok_end, mis2 = _cidx2widx(char_end, end_map, 1)
    if mis1 or mis2:
        zwarn(f"Problem of char idx: '{text[char_start:char_end]}' vs. '{toks[tok_start:tok_end+1]}'")
    assert tok_end >= tok_start
    return tok_start, tok_end-tok_start+1  # widx, wlen

def read_fulltext(file: str):
    tree = ef_parse(file)
    node_ft = tree.getroot()
    assert_tag(node_ft, "fullTextAnnotation")
    node_chs_map = get_subnodes_map(node_ft, lambda x: x.tag.split("}")[-1])
    assert len(node_chs_map["header"]) == 1
    node_corpus = get_subnodes_by(node_chs_map["header"][0], lambda n: n.tag.endswith("corpus"), True)
    node_document = get_subnodes_by(node_corpus, lambda n: n.tag.endswith("document"), True)
    corpus_name, document_name = node_corpus.get('name'), node_document.get('name')
    if document_name is None:
        document_name = node_document.get('description')
    assert document_name is not None
    doc_id = f"{corpus_name}__{document_name}"
    assert os.path.basename(file) == doc_id+".xml"
    ret_doc = Doc.create(id=doc_id)
    for node_sent in node_chs_map["sentence"]:
        sent = _build_sent(node_sent)
        sent.info["paragNo"] = int(node_sent.get("paragNo"))
        ret_doc.add_sent(sent)
    return ret_doc

# =====

class FnReader:
    def __init__(self, fn_dir: str):
        self.fn_dir = fn_dir
        # --
        self.load_indexes()

    def load_indexes(self):
        # -----
        # read all the indexes
        zlog(f"Read all the indexes from dir {self.fn_dir}")
        self.frame_idxes = read_frame_idxes(f"{self.fn_dir}/frameIndex.xml")
        self.frame_relations = read_frame_relations(f"{self.fn_dir}/frRelation.xml")
        self.lu_idxes = read_lu_idxes(f"{self.fn_dir}/luIndex.xml")
        self.sem_types = read_sem_types(f"{self.fn_dir}/semTypes.xml")
        self.fulltext_idxes = read_fulltext_idxes(f"{self.fn_dir}/fulltextIndex.xml")
        # further build reverse one
        self.lu_map = {}
        for lu in self.lu_idxes:
            fname, luname = lu["frameName"], lu["name"]
            if fname not in self.lu_map:
                self.lu_map[fname] = {}
            if luname in self.lu_map[fname]:
                # todo(note): in fn15, there are repeated ones
                zwarn(f"Problem repeated lu entry: {fname}/{luname}")
                continue
            # assert luname not in self.lu_map[fname]
            self.lu_map[fname][luname] = lu

    def load_frame(self, name: str):
        return read_frame(f"{self.fn_dir}/frame/{name}.xml")

    def load_doc(self, name: str):
        return read_fulltext(f"{self.fn_dir}/fulltext/{name}.xml")

    def load_lu(self, fname: str, luname: str):
        lu = self.lu_map[fname][luname]
        return read_lu(f"{self.fn_dir}/lu/lu{lu['ID']}.xml")

    # =====
    def load_all_frames(self):
        res = {}
        for one_frame in self.frame_idxes:
            zlog(f"Load frame {one_frame}")
            fname = one_frame["name"]
            one_res = self.load_frame(fname)
            assert fname not in res
            res[fname] = one_res
        # collect stat
        c = Counter()
        c["all_frame"] = len(res)
        c["all_FE"] = sum(len(z['FE']) for z in res.values())
        c["all_lu"] = sum(len(z['lexUnit']) for z in res.values())
        c["all_FE_name"] = len(set(z2['name'] for z in res.values() for z2 in z['FE']))
        c["all_lu_name"] = len(set(z2['name'].split(".")[-2] for z in res.values() for z2 in z['lexUnit']))
        zlog(f"Load all frames: {c}")
        return res

    def load_all_lus(self):
        res = {}
        for one_lu in self.lu_idxes:
            zlog(f"Load LU {one_lu}")
            fname, luname = one_lu["frameName"], one_lu["name"]
            one_res = self.load_lu(fname, luname)
            if (fname, luname) in res:
                # todo(note): in fn15, there are repeated ones
                zwarn(f"Problem repeated lu entry: {fname}/{luname}")
                continue
            res[(fname, luname)] = one_res
            # todo(note): fn15 is not accurate for these
            # if (one_lu["numAnnotInstances"]>0) != one_lu["hasAnnotation"]:
            #     zwarn(f"Problem of hasAnnotation: {one_lu}")
            # if len(one_res["exemplars"]) != one_lu["numAnnotInstances"]:
            #     zwarn(f"Problem of unequal exemplars number: {len(one_res['exemplars'])} vs. {one_lu['numAnnotInstances']}")
        # --
        # stat
        c = Counter()
        c["all_lu"] = len(res)
        c["all_lu>0"] = sum(int(len(z["exemplars"]))>0 for z in res.values())
        c["all_exemplars"] = sum(len(z["exemplars"]) for z in res.values())
        zlog(f"Load all LU: {c}")
        return res

    def load_all_fulltexts(self):
        res = {}
        for one_ft in self.fulltext_idxes:
            zlog(f"Load doc {one_ft}")
            doc = self.load_doc(one_ft["name"])
            assert one_ft["name"] not in res
            res[one_ft["name"]] = doc
        # --
        # stat
        c = Counter()
        c["all_doc"] = len(res)
        c["all_sent"] = sum(len(z.sents) for z in res.values())
        c["all_toks"] = sum(len(z2) for z in res.values() for z2 in z.sents)
        c["all_toks_notNT"] = sum((z3!="NT") for z in res.values() for z2 in z.sents for z3 in z2.wsl)
        c["all_evt"] = sum(len(z2.events) for z in res.values() for z2 in z.sents)
        c["all_args"] = sum(len(z3.args) for z in res.values() for z2 in z.sents for z3 in z2.events)
        c["all_iargs"] = sum(len(z3.iargs) for z in res.values() for z2 in z.sents for z3 in z2.events)
        zlog(f"Load all fulltexts: {c}")
        return res

# =====
def main(fn_dir, output_dir):
    reader = FnReader(fn_dir)
    # testing
    # f = reader.load_frame("Change_position_on_a_scale")
    # doc = reader.load_doc("ANC__journal_christine")
    # lu = reader.load_lu("Change_position_on_a_scale", "fall.v")
    # testing loading all
    system(f"mkdir -p {output_dir}", pp=True)
    res1 = reader.load_all_frames()
    default_json_serializer.to_file(res1, f"{output_dir}/frames.json")
    res2 = reader.load_all_fulltexts()
    with DataWriter(get_text_dumper(f"{output_dir}/fulltext.json")) as writer:
        writer.write_insts([res2[k] for k in sorted(res2.keys())])
    res3 = reader.load_all_lus()
    with DataWriter(get_text_dumper(f"{output_dir}/exemplars.json")) as writer:
        writer.write_insts(z for k in sorted(res3.keys()) for z in res3[k]["exemplars"])
    # breakpoint()

# python3 fn_reader.py ?
if __name__ == '__main__':
    main(*sys.argv[1:])

# PYTHONPATH=../../src/ python3 fn_reader.py ~/nltk_data/corpora/framenet_v15 fn15 |& tee _log.fn15
# PYTHONPATH=../../src/ python3 fn_reader.py ~/nltk_data/corpora/framenet_v17 fn17 |& tee _log.fn17
