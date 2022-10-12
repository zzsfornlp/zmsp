#

# step 1: extract
# -- this step simply extracts all information from raw files, only do extracting and keeping all raw information!!

# data format
"""
Doc = {
    # basic info
    "id": str, "dataset": str, "text": str, "info": Dict,
    # sentence
    "sents": List[Sent],
    # mentions
    "ef_mentions"/"rel_mentions"/"evt_mentions": List[Ef/Rel/Evt],
    # clusters
    "ef_clusters"/"rel_clusters"/"evt_clusters": List[Cluster],
},
Ef = {"id": str, "posi": Posi, "type": str, "info": Dict},
Rel = {"id": str, "posi": Posi, "type": str, "args": [{"aid": str, "role": str}], "info": Dict},
Evt = {"id": str, "posi": Posi, "type": str, "args": [{"aid": str, "role": str}], "info": Dict},
Cluster = {"ids": List[str], "info": Dict},
Posi = {"head": Posi, "posi_char": (cidx, clen), "posi_token": (sid, widx, wlen)},
Sent = {"tokens": List[str], "positions": List[(cidx, clen)], ...}
"""

from typing import List
import os, sys, re
import json
import xml.etree.ElementTree as ET
import re

# =====
# helpers
def zlog(s):
    print(str(s), file=sys.stderr, flush=True)

def zopen(filename, mode='r', encoding="utf-8"):
    if 'b' in mode:
        encoding = None
    return open(filename, mode, encoding=encoding)

def zwarn(s):
    zlog("Warning: " + str(s))

def get_fname_id(fname: str, exclude_endings, include_endings):
    if fname[0] == ".":
        return None
    for one_end in exclude_endings:
        if fname.endswith(one_end):
            return None
    for one_end in include_endings:
        if fname.endswith(one_end):
            return fname[:-len(one_end)]
    return None

# --
# make new items
def new_doc(id: str, dataset: str, text: str, info: dict):
    return {
        "id": id, "dataset": dataset, "text": text, "info": info,
        "sents": None,
        "ef_mentions": [], "rel_mentions": [], "evt_mentions": [],
        "ef_clusters": [], "rel_clusters": [], "evt_clusters": [],
    }

def new_cluster(ids: list, info: dict):
    return {"ids": ids, "info": info}

def new_mention(id: str, posi: dict, type: str, info: dict):
    return {"id": id, "posi": posi, "type": type, "info": info}

def new_posi(cidx: int, clen: int, head_cidx=None, head_clen=None):
    ret = {"posi_char": (cidx, clen)}
    if head_cidx is not None and head_clen is not None:
        ret["head"] = {"posi_char": (head_cidx, head_clen)}
    return ret

# =====
# read from ACE2005 format

# read from ace
def read_ace_doc(dataset_name: str, doc_id_from_file: str, file_apf: str, file_source: str):
    # zlog(f"Read doc {dataset_name} {doc_id_from_file}")
    assert os.path.basename(file_apf).startswith(doc_id_from_file)
    assert os.path.basename(file_source).startswith(doc_id_from_file)
    # ===== read annotation: specifically do it
    with zopen(file_source) as fd:
        source_str = fd.read()
    notag_source_str = re.compile('<.*?>', re.DOTALL).sub("", source_str)
    # notag_source_str = ""
    # tag_open = False
    # _TAG_SET = set("ABCDEFGHIJKLMNOPQRSTUVWXYX")
    # for ii, cc in enumerate(source_str):
    #     if tag_open:  # to avoid some in-between "<" or ">"
    #         if cc == '>' and ii>0 and (source_str[ii-1] in _TAG_SET or source_str[ii-1]=='"'):
    #             tag_open = False
    #     else:
    #         if cc == '<' and ii+1<len(source_str) and (source_str[ii+1] in _TAG_SET or source_str[ii+1]=='/'):
    #             tag_open = True
    #         else:
    #             notag_source_str += cc
    # =====
    doc = new_doc(doc_id_from_file, dataset_name, source_str, {})
    doc_tree = ET.parse(file_apf)
    doc_node = doc_tree.getroot()[0]  # <document>
    assert doc_node.tag == "document"
    assert doc_node.get("DOCID") == doc_id_from_file
    # --
    def _parse_char_node(_char_node):  # parse char position from mention node
        _moff = int(_char_node.get("START"))
        _mlen = int(_char_node.get("END")) + 1 - _moff
        # check str match
        _src_text = re.sub("&amp;", "&", notag_source_str[_moff:_moff+_mlen])
        _src_text = re.sub("&#8226;", "•", _src_text)
        # assert _char_node.text == _src_text
        if _char_node.text != _src_text:
            zwarn(f"Text mismatch @{doc_id_from_file}: {_char_node.text} vs. {_src_text}")
        return _moff, _mlen
    # --
    # read each elements
    for one_item_node in doc_node:
        # --
        tag = one_item_node.tag
        cur_mentions = []
        # --
        if tag in ("entity", "value", "timex2"):
            cur_prefix = "ef"
            is_entity, is_time = (tag == "entity"), (tag == "timex2")
            # make the type
            if is_time:
                vtype = "Time"
            elif "SUBTYPE" in one_item_node.attrib:
                vtype = f"{one_item_node.get('TYPE')}.{one_item_node.get('SUBTYPE')}"
            else:
                vtype = one_item_node.get('TYPE')
            # --
            for one_mention in one_item_node.findall(tag + "_mention"):
                mid = one_mention.get("ID")
                moff, mlen = _parse_char_node(one_mention.find("extent").find("charseq"))
                extra_info = {"is_entity": is_entity}
                if is_entity:
                    extra_info["mtype"] = one_mention.get("TYPE")  # mention level entity type
                    head_moff, head_mlen = _parse_char_node(one_mention.find("head").find("charseq"))  # entity has head
                else:
                    head_moff = head_mlen = None
                # make a new mention
                cur_ef = new_mention(mid, new_posi(moff, mlen, head_moff, head_mlen), vtype, extra_info)
                cur_mentions.append(cur_ef)
            # --
        elif tag in ("relation", "event"):
            cur_prefix = "rel" if tag=="relation" else "evt"
            cur_posi_name = "extent" if tag=="relation" else "anchor"
            cur_type = f"{one_item_node.get('TYPE')}.{one_item_node.get('SUBTYPE')}"
            for one_mention in one_item_node.findall(tag + "_mention"):
                mid = one_mention.get("ID")
                moff, mlen = _parse_char_node(one_mention.find(cur_posi_name).find("charseq"))
                # make a new mention
                cur_one = new_mention(mid, new_posi(moff, mlen), cur_type, {})
                cur_mentions.append(cur_one)
                # add args
                cur_one["args"] = [{"aid": z.get("REFID"), "role": z.get("ROLE")}
                                   for z in one_mention.findall(f"{tag}_mention_argument")]
            # --
        else:
            raise RuntimeError(f"UNK tag: {tag}")
        # --
        # append all and make a new cluster
        doc[f"{cur_prefix}_mentions"].extend(cur_mentions)
        doc[f"{cur_prefix}_clusters"].append(new_cluster([z["id"] for z in cur_mentions], {}))
    # --
    return doc

# read ACE dataset
def read_ace_dataset(dataset_name: str, dir_apf: str, dir_source: str):
    # =====
    def _get_apf_id(fname: str):
        return get_fname_id(fname, [], [".apf.xml"])
    def _get_src_id(fname: str):
        return get_fname_id(fname, [], [".sgm"])
    # =====
    zlog(f"Read dataset {dataset_name}")
    # based on the source files
    docs = []
    for sfile in sorted(os.listdir(dir_source)):
        cur_doc_id = _get_src_id(sfile)
        if cur_doc_id == "ALFILFILM_20050202.0740":
            continue  # note: ignore this ar doc, seems to have offset problem??
        if cur_doc_id is not None:
            full_sfile = os.path.join(dir_source, sfile)
            full_afile = os.path.join(dir_apf, cur_doc_id+".apf.xml")
            assert os.path.exists(full_afile)
            cur_doc = read_ace_doc(dataset_name, cur_doc_id, full_afile, full_sfile)
            docs.append(cur_doc)
    zlog(f"End reading dataset {dataset_name}: {len(docs)} docs.")
    return docs


# =====
# read from ERE format:

# read one doc from ere and source
def read_ere_doc(dataset_name: str, doc_id_from_file: str, file_eres: List[str], file_source: str):
    # zlog(f"Read doc {dataset_name} {doc_id_from_file}")
    # first some checkings
    if not isinstance(file_eres, (list, tuple)):
        file_eres = [file_eres]
    assert all(os.path.basename(z).startswith(doc_id_from_file) for z in file_eres + [file_source])
    # collect the source string
    with zopen(file_source) as fd:
        source_str = fd.read()
    # --
    doc = new_doc(doc_id_from_file, dataset_name, source_str, {})
    for file_ere in file_eres:
        doc_tree = ET.parse(file_ere)
        doc_node = doc_tree.getroot()  # <deft_ere>
        assert doc_node.tag == "deft_ere"
        assert doc_node.get("doc_id").startswith(doc_id_from_file)  # sometimes id can have more
        # --
        def _parse_char_node(_char_node):  # parse char position from mention node
            # --
            _trigger_node = list(_char_node.findall("trigger"))
            if len(_trigger_node)>0:
                assert len(_trigger_node)==1 and "offset" not in _char_node.attrib and "length" not in _char_node.attrib
                _char_node = _trigger_node
            if _char_node.find("mention_text") is None:
                _mention_text = _char_node.text  # directly text
            else:
                _mention_text = _char_node.find("mention_text").text
            # --
            _moff = int(_char_node.get("offset"))
            _mlen = int(_char_node.get("length"))
            # check str match
            _s1, _s2 = ' '.join(_mention_text.split()), ' '.join(source_str[_moff:_moff+_mlen].split())
            s1sub = re.sub("&amp;", "&", _s1)
            s1sub = re.sub("&gt;", ">", s1sub)
            s1sub = re.sub("&lt;", "<", s1sub)
            assert s1sub == _s2 or _s1 == _s2
            return _moff, _mlen
        # --
        def _get_aid(_arg):
            if 'entity_mention_id' in _arg.attrib:
                return _arg.get('entity_mention_id')
            else:
                assert 'filler_id' in _arg.attrib
                return _arg.get('filler_id')
        # --
        # read them all
        tag_info = {"entities": ("entity", ), "fillers": ("filler", ),
                    "relations": ("relation", ), "hoppers": ("hopper", )}
        for one_set_node in doc_node:
            tag0 = one_set_node.tag
            tag, = tag_info[tag0]
            del tag_info[tag0]  # no repeating!!
            # further read
            for one_item_node in one_set_node:
                # --
                assert one_item_node.tag == tag
                cur_mentions = []
                # --
                if tag == "entity":
                    cur_prefix = "ef"
                    etype = one_item_node.get("type")
                    # --
                    for one_mention in one_item_node.findall(tag + "_mention"):
                        mid = one_mention.get("id")
                        moff, mlen = _parse_char_node(one_mention)
                        extra_info = {"is_entity": True, "mtype": one_mention.get("noun_type")}
                        if extra_info["mtype"] == "NOM":
                            assert one_mention.find("nom_head") is not None
                        if one_mention.find("nom_head") is not None:
                            head_moff, head_mlen = _parse_char_node(one_mention.find("nom_head"))  # entity has head
                        else:
                            head_moff = head_mlen = None
                        # make a new mention
                        cur_ef = new_mention(mid, new_posi(moff, mlen, head_moff, head_mlen), etype, extra_info)
                        cur_mentions.append(cur_ef)
                    # --
                elif tag == "filler":
                    cur_prefix = "ef"
                    one_mention = one_item_node  # directly mention itself
                    moff, mlen = _parse_char_node(one_mention)
                    # make a new mention
                    cur_ef = new_mention(one_mention.get("id"), new_posi(moff, mlen, None, None),
                                         one_mention.get("type"), {"is_entity": False})
                    cur_mentions.append(cur_ef)
                elif tag == "relation":
                    cur_prefix = "rel"
                    cur_type = f"{one_item_node.get('type')}.{one_item_node.get('subtype')}"
                    for one_mention in one_item_node.findall("relation_mention"):
                        mid = one_mention.get("id")
                        if one_mention.find("trigger") is not None:
                            moff, mlen = _parse_char_node(one_mention.find("trigger"))
                            cur_posi = new_posi(moff, mlen)
                        else:
                            cur_posi = None  # no trigger
                        # make a new mention
                        cur_one = new_mention(mid, cur_posi, cur_type, {"realis": one_mention.get('realis')})
                        cur_mentions.append(cur_one)
                        # add args
                        cur_one["args"] = [{"aid": _get_aid(z), 'role': z.get('role')} for z in one_mention if z.tag.startswith("rel_arg")]
                    # --
                elif tag == "hopper":
                    cur_prefix = "evt"
                    for one_mention in one_item_node.findall("event_mention"):
                        mid = one_mention.get("id")
                        cur_type = f"{one_mention.get('type')}.{one_mention.get('subtype')}"  # type@mention
                        moff, mlen = _parse_char_node(one_mention.find("trigger"))
                        # make a new mention
                        cur_one = new_mention(mid, new_posi(moff, mlen), cur_type, {"realis": one_mention.get('realis')})
                        cur_mentions.append(cur_one)
                        # add args
                        cur_one["args"] = [{"aid": _get_aid(z), 'role': z.get('role')} for z in one_mention.findall("em_arg")]
                    # --
                else:
                    raise RuntimeError(f"UNK tag: {tag}")
                # --
                # append all and make a new cluster
                doc[f"{cur_prefix}_mentions"].extend(cur_mentions)
                doc[f"{cur_prefix}_clusters"].append(new_cluster([z["id"] for z in cur_mentions], {}))
            # --
        # --
    # --
    return doc

# read all ere and sources within a dataset
def read_ere_dataset(dataset_name: str, dir_ere: str, dir_source: str):
    # =====
    def _get_ere_id(fname: str):
        cur_doc_id = get_fname_id(fname, [], [".rich_ere.xml"])
        if cur_doc_id is not None:
            name_split = cur_doc_id.split("_")
            if "-" in name_split[-1]:
                cur_doc_id = "_".join(name_split[:-1])
        return cur_doc_id

    def _get_src_id(fname: str):
        return get_fname_id(fname, [".rich_ere.xml"], [".mpdf.xml", ".xml", ".cmp.txt", ".mp.txt", ".txt"])
    # =====
    # zlog(f"Read dataset {dataset_name}")
    # first get all source files
    all_source_files = []
    source2eres = {}
    for file in sorted(os.listdir(dir_source)):
        cur_doc_id = _get_src_id(file)
        if cur_doc_id is not None:
            all_source_files.append((file, cur_doc_id))
            assert cur_doc_id not in source2eres
            source2eres[cur_doc_id] = []
    # then put in all ere files
    for file in sorted(os.listdir(dir_ere)):
        cur_doc_id = _get_ere_id(file)
        if cur_doc_id is not None:
            assert cur_doc_id in source2eres
            source2eres[cur_doc_id].append(file)
    # based on source
    docs = []
    for one_source_file, one_doc_id in all_source_files:
        cur_source_file = os.path.join(dir_source, one_source_file)
        cur_ere_files = [os.path.join(dir_ere, z) for z in source2eres[one_doc_id]]
        if len(cur_ere_files) == 0:
            zwarn(f"No ere file for {cur_source_file}")
            continue
        if len(cur_ere_files) > 1:
            # zlog(f"Multiple ere files for one source: {cur_source_file} {cur_ere_files}")
            pass
        cur_doc = read_ere_doc(dataset_name, one_doc_id, cur_ere_files, cur_source_file)
        docs.append(cur_doc)
    zlog(f"End reading dataset {dataset_name}: {len(docs)} docs.")
    return docs

# ======
# read kbp15 format (nugget & hopper)
def read_kbp15_doc(dataset_name: str, doc_id_from_file: str, file_nugget: str, file_hopper: str, file_source: str):
    # zlog(f"Read doc {dataset_name} {doc_id_from_file}")
    assert os.path.basename(file_nugget).startswith(doc_id_from_file)
    assert os.path.basename(file_hopper).startswith(doc_id_from_file)
    assert os.path.basename(file_source).startswith(doc_id_from_file)
    # =====
    # read source
    with zopen(file_source) as fd:
        source_str = fd.read()
    # =====
    # helpers
    def _parse_mention(one_mention):
        assert one_mention.tag == "event_mention"
        mid = one_mention.get("id")
        cur_type = f"{one_mention.get('type')}.{one_mention.get('subtype')}"  # type@mention
        trigger_node = one_mention.find("trigger")
        moff, mlen = int(trigger_node.get("offset")), int(trigger_node.get("length"))
        assert ' '.join(trigger_node.text.split()) == ' '.join(source_str[moff:moff+mlen].split())
        # make a new mention
        v_mention = new_mention(mid, new_posi(moff, mlen), cur_type, {"realis": one_mention.get('realis')})
        v_mention["args"] = []  # make it consistent, actually no annotations for args
        return v_mention
    def _parse_hopper(one_hopper):
        assert one_hopper.tag == "hopper"
        v_hopper = new_cluster([_parse_mention(z)['id'] for z in one_hopper], {})
        return v_hopper
    # =====
    # read nuggets
    nugget_tree = ET.parse(file_nugget)
    nugget_root = nugget_tree.getroot()  # <document>
    assert nugget_root.get("doc_id") == doc_id_from_file
    mentions = [_parse_mention(node) for node in nugget_root]
    # read hoppers
    hopper_tree = ET.parse(file_hopper)
    hopper_root = hopper_tree.getroot()
    assert hopper_root.get("doc_id") == doc_id_from_file
    assert len(hopper_root)==1 and hopper_root[0].tag == "hoppers"
    hoppers = [_parse_hopper(node) for node in hopper_root[0]]
    # =====
    # put them together
    assert set(z['id'] for z in mentions) == set(sum([z['ids'] for z in hoppers], []))
    doc = new_doc(doc_id_from_file, dataset_name, source_str, {})
    doc["evt_mentions"].extend(mentions)
    doc["evt_clusters"].extend(hoppers)
    return doc

# read kbp-15 EN dataset
def read_kbp15_dataset(dataset_name: str, dir_nugget: str, dir_hopper: str, dir_source: str):
    # =====
    def _get_src_id(fname: str):
        return get_fname_id(fname, [], [".txt"])
    # =====
    zlog(f"Read dataset {dataset_name}")
    # based on source file
    docs = []
    for sfile in sorted(os.listdir(dir_source)):
        cur_doc_id = _get_src_id(sfile)
        if cur_doc_id is not None:
            full_nfile = os.path.join(dir_nugget, cur_doc_id+".event_nuggets.xml")
            full_hfile = os.path.join(dir_hopper, cur_doc_id+".event_hoppers.xml")
            full_sfile = os.path.join(dir_source, sfile)
            cur_doc = read_kbp15_doc(dataset_name, cur_doc_id, full_nfile, full_hfile, full_sfile)
            docs.append(cur_doc)
    zlog(f"End reading dataset {dataset_name}: {len(docs)} docs.")
    return docs

# =====
# outmost ones

# write
def write_dataset(docs: list, output_fname: str):
    with zopen(output_fname, "w") as fd:
        zlog(f"Write dataset to {output_fname} (N={len(docs)})")
        for doc in docs:
            fd.write(json.dumps(doc, ensure_ascii=False) + "\n")
        # --

# reads
def read_ace(data_dir: str, language: str, output_file: str, **kwargs):
    info_table = {
        "en": ("English", "bc bn cts nw un wl", "timex2norm"),
        "zh": ("Chinese", "bn nw wl", "adj"),
        "ar": ("Arabic", "bn nw wl", "adj"),
    }
    # --
    outer_dir, mid_dirs, inner_dir = info_table[language]
    main_dir = os.path.join(data_dir, "LDC2006T06", "ace_2005_td_v7", "data", outer_dir)
    # read
    all_docs = []
    for set_name in mid_dirs.split():
        dataset_name = f"{language}.ace."+set_name
        dataset_dir = os.path.join(main_dir, set_name, inner_dir)
        one_docs = read_ace_dataset(dataset_name, dataset_dir, dataset_dir)
        all_docs.extend(one_docs)
    # write
    if output_file:
        write_dataset(all_docs, output_file)
    return all_docs

def read_ere(data_dir: str, language: str, output_file: str, **kwargs):
    info_table = {
        "en": (
            [
                ("LDC2015E29", "data/ere/mpdfxml/", "data/source/mpdfxml/"),
                ("LDC2015E68", "data/ere/", "data/source/"),
                ("LDC2016E31", "data/ere/", "data/source/"),
                # this is usually not used, skip
                # ("LDC2015E78", "data/eng/ere/", "data/eng/translation/"),
                ("LDC2016E73.df", "data/eng/df/ere/", "data/eng/df/source/"),
                ("LDC2016E73.nw", "data/eng/nw/ere/", "data/eng/nw/source/"),
                ("LDC2017E54.df", "data/eng/df/ere/", "data/eng/df/source/"),
                ("LDC2017E54.nw", "data/eng/nw/ere/", "data/eng/nw/source/"),
                # E55 extends args, currently skip
                # ("LDC2017E55.df", "data/eng/df/ere/", "data/eng/df/source/"),
                # ("LDC2017E55.nw", "data/eng/nw/ere/", "data/eng/nw/source/"),
            ],
        ),
        "zh": (
            [
                ("LDC2015E105", "data/ere/", "data/source/"),
                ("LDC2015E112", "data/ere/", "data/source/"),
                ("LDC2015E78", "data/cmn/ere/", "data/cmn/source/"),
                ("LDC2016E73.df", "data/cmn/df/ere/", "data/cmn/df/source/"),
                ("LDC2016E73.nw", "data/cmn/nw/ere/", "data/cmn/nw/source/"),
                ("LDC2017E54.df", "data/cmn/df/ere/", "data/cmn/df/source/"),
                ("LDC2017E54.nw", "data/cmn/nw/ere/", "data/cmn/nw/source/"),
                # E55 extends args, currently skip
                # ("LDC2017E55.df", "data/cmn/df/ere/", "data/cmn/df/source/"),
                # ("LDC2017E55.nw", "data/cmn/nw/ere/", "data/cmn/nw/source/"),
            ],
        ),
        "es": (
            [
                ("LDC2015E107", "data/ere/", "data/source/"),
                ("LDC2016E34", "data/ere/", "data/source/"),
                ("LDC2016E73.df", "data/spa/df/ere/", "data/spa/df/source/"),
                ("LDC2016E73.nw", "data/spa/nw/ere/", "data/spa/nw/source/"),
                ("LDC2017E54.df", "data/spa/df/ere/", "data/spa/df/source/"),
                ("LDC2017E54.nw", "data/spa/nw/ere/", "data/spa/nw/source/"),
                # E55 extends args, currently skip
                # ("LDC2017E55.df", "data/spa/df/ere/", "data/spa/df/source/"),
                # ("LDC2017E55.nw", "data/spa/nw/ere/", "data/spa/nw/source/"),
            ],
        ),
    }
    # --
    ere_data, = info_table[language]
    # read
    all_docs = []
    for dataset_name, ere_dir, source_path in ere_data:
        dir_name = dataset_name.split(".")[0]  # use simplified dir-name
        one_docs = read_ere_dataset(f"{language}.ere."+dataset_name, os.path.join(data_dir, dir_name, ere_dir),
                                    os.path.join(data_dir, dir_name, source_path))
        all_docs.extend(one_docs)
    # write
    if output_file:
        write_dataset(all_docs, output_file)
    return all_docs

def read_kbp15(data_dir: str, language: str, output_file: str, **kwargs):
    assert language == "en"
    # extra EN15 data (no arg)
    main_dir = os.path.join(data_dir, "LDC2017E02", "data", "2015")
    all_docs = []
    for set_name, nugget_dir, hopper_dir in zip(["eval", "training"], ["nugget", "event_nugget"],
                                                ["hopper", "event_hopper"]):
        dataset_name = "en.kbp15." + set_name
        one_docs = read_kbp15_dataset(dataset_name, os.path.join(main_dir, set_name, nugget_dir),
                                      os.path.join(main_dir, set_name, hopper_dir),
                                      os.path.join(main_dir, set_name, "source"))
        all_docs.extend(one_docs)
    # write
    if output_file:
        write_dataset(all_docs, output_file)
    return all_docs

# --
def main(data_dir: str, code: str, output_file: str):
    language, dset = code.split('.')
    read_f = globals()[f"read_{dset}"]
    read_f(data_dir, language, output_file)
    # --

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

# --
# for cc in en.ace en.ere en.kbp15 zh.ace zh.ere es.ere ar.ace; do python3 s1_extract.py ./raw/ ${cc} ${cc}.json; done
