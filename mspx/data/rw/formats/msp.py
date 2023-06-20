#

# from previous old msp versions

__all__ = [
    'Msp2Formator',
]

import json
from mspx.data.inst import Doc, Sent, DataInst
from mspx.utils import zwarn
from .base import DataFormator

@DataFormator.rd('msp2')
class Msp2Formator(DataFormator):
    # in-place change key
    def _change_key(self, obj, d: dict, stop_set: set):
        if isinstance(obj, dict):
            for k in list(obj.keys()):
                if k in stop_set:
                    continue
                v = obj[k]
                del obj[k]
                obj[d.get(k, k)] = self._change_key(v, d, stop_set)
            return obj
        elif isinstance(obj, list):
            return [self._change_key(v, d, stop_set) for v in obj]
        else:
            return obj

    def to_obj(self, inst: Doc):
        for sent in inst.sents:  # especially set sent's id
            sent.set_id(f's{sent.sid}')
        # --
        dx = inst.to_dict(store_type=False)
        self._change_key(dx, {'id': '_id'}, {'info'})  # id -> _id
        if '_frames' in dx:  # no doc-level frames
            del dx['_frames']
        # add sent-level ones
        list_items = []
        dict_items = {}  # id(item) -> dict (stored to add args later)
        for sent in inst.sents:
            sent_dict = dx['sents'][sent.sid]
            for cate, key in zip(['ef', 'evt'], ['entity_fillers', 'events']):
                _ds = []
                for item in sent.yield_frames(cates=cate):
                    v0 = item.to_dict(store_type=False)
                    self._change_key(v0, {'id': '_id'}, {'info'})  # id -> _id
                    if isinstance(v0.get('mention'), dict) and '_sid' in v0['mention']:
                        del v0['mention']['_sid']  # note: no need this!
                    if 'label' in v0:
                        v0['type'] = v0['label']
                        del v0['label']
                    if 'args' in v0:
                        del v0['args']
                    _ds.append(v0)
                    list_items.append(item)
                    assert id(item) not in dict_items
                    dict_items[id(item)] = v0
                sent_dict[key] = _ds
        # add args
        for item in list_items:
            v0 = dict_items[id(item)]
            _args = []
            for ii, arg in enumerate(item.args):
                if id(arg.arg) not in dict_items:
                    zwarn(f"ArgLink ref to un-added item: {arg}")
                    continue  # ignore this one!
                # --
                # add args
                va = arg.to_dict(store_type=False)
                del va['_arg']
                va['main'] = '..'
                va['arg'] = arg.arg.id if arg.arg.sent is item.sent else f"s{arg.arg.sent.sid}/{arg.arg.id}"
                if 'label' in va:
                    va['role'] = va['label']
                    del va['label']
                va['_id'] = f"arg{ii}"
                _args.append(va)
                # --
                # add to as_args
                v1 = dict_items[id(arg.arg)]
                if 'as_args' not in v1:
                    v1['as_args'] = []
                v1['as_args'].append((item.id if arg.arg.sent is item.sent else f"s{item.sent.sid}/{item.id}")
                                     + f"/arg{ii}")
                # --
            v0['args'] = _args
        # --
        return json.dumps(dx, ensure_ascii=False)

    def from_obj(self, s: str):
        d0 = json.loads(s)
        self._change_key(d0, {'_id': 'id'}, {'info'})  # _id -> id
        if 'sents' not in d0:  # wrap for one-sent doc!
            d0 = {'sents': [d0]}
        # --
        sent_items = {}  # sid -> {eid -> D}
        for d_sent in d0['sents']:
            d_sid = d_sent['id']
            sent_items[d_sid] = {}
            for cate, key in zip(['ef', 'evt'], ['entity_fillers', 'events']):
                if key in d_sent:
                    for item in d_sent[key]:
                        item['_cate'] = cate
                        d_eid = item['id']
                        assert d_eid not in sent_items[d_sid]
                        sent_items[d_sid][d_eid] = item
                    del d_sent[key]
        # --
        doc = Doc.create_from_dict(d0)
        # add frames
        for sent in doc.sents:
            d_sid = sent.id
            for eid in list(sent_items[d_sid].keys()):
                item = sent_items[d_sid][eid]
                frame = sent.make_frame(item['mention']['widx'], item['mention']['wlen'],
                                        label=item.get('type'), cate=item['_cate'], score=item.get('score'))
                frame.mention.from_dict(item['mention'])
                if 'info' in item:
                    frame.info.update(item['info'])
                frame.args = item.get('args', [])  # note: not yet!
                sent_items[d_sid][eid] = frame  # simply replace!
        # add args
        for frame in doc.yield_frames():
            d_args = frame.args
            frame.args = []
            for da in d_args:
                _aref = da['arg']
                if '/' not in _aref:
                    a_item = sent_items[frame.sent.id][_aref]
                else:
                    _n1, _n2 = _aref.rsplit('/', 1)
                    a_item = sent_items[_n1][_n2]
                alink = frame.add_arg(a_item, label=da.get('role'), score=da.get('score'))
                if 'info' in da:
                    alink.info.update(da['info'])
        # --
        return doc

# --
# b mspx/data/rw/formats/msp:59
