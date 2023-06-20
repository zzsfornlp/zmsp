#

# utils for mongodb

import pandas
import re
import time
import numpy as np
from collections import Counter, OrderedDict
import os
from mspx.utils import Conf, zlog, zwarn, ZObject, zopen, auto_mkdir, init_everything, zglobs, default_pickle_serializer, ZHelper
from mspx.data.inst import Doc, Sent
from mspx.data.rw import ReaderGetterConf, WriterGetterConf

class MainConf(Conf):
    def __init__(self):
        self.cmd = ''  # task to do
        self.mdb = MdbHelperConf()

class MdbHelperConf(Conf):
    def __init__(self):
        self.uri = "mongodb://localhost:27017"
        self.passwd = "**"
        self.database = None  # default one with 'get_database'
        # --
        self.qcol = 'data_queue'  # queue collection name!
        self.col_ent = '_ent'  # col suffix for entity
        self.col_rel = '_rel'  # col suffix for relation
        self.id_sep = ':::'  # doc_id:::item_id

class MdbHelper:
    def __init__(self, conf: MdbHelperConf):
        self.conf = conf
        self.db = self.get_db(conf)  # obtain the db

    @staticmethod
    def get_db(conf: MdbHelperConf):
        from pymongo import MongoClient
        from getpass import getpass
        # get password
        _uri0, _pp = conf.uri, conf.passwd
        _uri = _uri0
        if all(c=='*' for c in _pp) and _pp in _uri:
            _pp_real = getpass(f"For connecting {_uri0}\nPassword: ")
            _uri = _uri.replace(_pp, _pp_real)
        # connect
        client = MongoClient(_uri)
        db = client.get_database(conf.database)
        zlog(f"Get db: {db}")
        return db

    def get_cols(self, col_re: str, get_one=False, ignore_special=False):
        conf: MdbHelperConf = self.conf
        db = self.db
        if str.isidentifier(col_re):  # just one!
            ret = [db[col_re]]
        else:
            ret = []
            pat = re.compile(col_re) if col_re else None
            for col_name in db.list_collection_names():
                if pat is not None and not pat.fullmatch(col_name):
                    continue  # require full match is specified!
                if ignore_special and any(col_name.endswith(z) for z in [conf.col_ent, conf.col_rel]):
                    continue
                ret.append(db[col_name])
        if get_one:
            assert len(ret) == 1
            return ret[0]
        else:
            return ret

    def get_f(self, eval_f, args):
        return ZHelper.eval_ff(eval_f, args, locals=locals().copy(), globals=globals().copy()) if eval_f else None

    def yield_items(self, col, cc, yield_col=False, ff='', qff='', quiet=False):
        ff = self.get_f(ff, 'x')
        qff = eval(qff) if qff else None
        for one_col in self.get_cols(col, False):
            cc['col'] += 1
            if not quiet:
                zlog(f"# With {one_col}:")
            for item in one_col.find(qff):
                if ff and not ff(item):
                    continue
                cc['item'] += 1
                if yield_col:
                    yield one_col, item
                else:
                    yield item

    def check_string_match(self, x0, x1, ignore_space=True):
        s0 = x0 if isinstance(x0, str) else ' '.join(x0)
        s1 = x1 if isinstance(x1, str) else ' '.join(x1)
        if ignore_space:
            s0 = ''.join(s0.split())
            s1 = ''.join(s1.split())
        assert s0 == s1, f"Mismatched strings: {x0} vs {x1}"

    def run(self, cmd: str, *args, **kwargs):
        from mspx.tools.utils import CmdLineParser
        cmd, target, r_args, r_kwargs = CmdLineParser.parse_cmd(cmd, assign_target=False)
        r_args.extend(args)
        r_kwargs.update(kwargs)
        cc = getattr(self, 'run_'+r_args[0])(*r_args[1:], **r_kwargs)
        zlog(f"Run {(r_args, r_kwargs)}: {ZHelper.resort_dict(cc)}")
        return cc

    # list all cols
    def run_list(self, col='', ff='', qff=''):
        cc = Counter()
        for item in self.yield_items(col, cc, ff=ff, qff=qff):
            zlog(item)
        return cc

    # queue utils
    def run_qutils(self, data=None, new_status=None):
        conf: MdbHelperConf = self.conf
        col = self.db[conf.qcol]
        query = {'data': data} if data else {}
        query['status'] = {'$ne': 'done'}
        one = col.find_one(query)
        ret = {}
        if one:  # find one
            data = one['data']
            old_status = one.get('status')
            if new_status:
                if 'logs' not in one:
                    one['logs'] = []
                one['logs'].append([f"{time.ctime()} {time.strftime('%Z')}", old_status, new_status])
            if new_status:
                one['status'] = new_status
            col.replace_one({'_id': one['_id']}, one)
            zlog(f'qutils_res {data} {old_status} {new_status}')
            ret.update({'data': data, 'old_status': old_status, 'new_status': new_status})
        else:
            if new_status:
                zwarn(f"Cannot find entry for {data}!!")
        # --
        return ret

    # note: keep it simple, otherwise using external tools to handle!
    #  two modes: 1) raw text without ann, 2) tokenized with ann

    # read data
    def run_read(self, col, output_path: str, ff='', qff=''):
        conf: MdbHelperConf = self.conf
        cc = Counter()
        all_insts = []
        for col, dd in self.yield_items(col, cc, yield_col=True, ff=ff, qff=qff):
            inst = Doc(text=dd['text'], id=dd['doc_id'])
            cc['doc'] += 1
            if 'info' in dd:  # extra info
                inst.info.update(dd['info'])
            if 'sents' in dd:  # tokenized!
                cc['doc_T'] += 1
                for _tokens in dd['sents']:
                    sent = Sent(_tokens)
                    inst.add_sent(sent)
                    cc['sent'] += 1
                    cc['tok'] += len(_tokens)
            all_insts.append(inst)
            # ents and rels
            d_ents, d_rels = list(self.db[col.name+conf.col_ent].find({'doc_id': dd['doc_id']})), list(self.db[col.name+conf.col_rel].find({'doc_id': dd['doc_id']}))
            if len(d_ents) > 0:
                cc['ent'] += len(d_ents)
                cc['rel'] += len(d_rels)
                m_ents = {}  # id -> frame
                for ee in d_ents:
                    _cfid, _posi, _label = [ee[z] for z in ['frame_id', 'position', 'label']]
                    _cate, _fid = _cfid.split(':', 1)  # note: special frame_id
                    _frame = inst.sents[_posi[0]].make_frame(_posi[1], _posi[2], _label, _cate, id=_fid)
                    assert _cfid not in m_ents
                    m_ents[_cfid] = _frame
                    if 'info' in ee:
                        _frame.info.update(ee['info'])
                    # sanity check
                    self.check_string_match(ee['text'], _frame.mention.get_words())
                    self.check_string_match(ee['text_context'], _frame.sent.get_text())
                for rr in d_rels:
                    _fid0, _fid1, _label = [rr[z] for z in ['arg0', 'arg1', 'label']]
                    _frame0, _frame1 = m_ents[_fid0], m_ents[_fid1]
                    alink = _frame0.add_arg(_frame1, _label)
                    if 'info' in rr:
                        alink.info.update(rr['info'])
                    # sanity check
                    self.check_string_match(rr['text_arg0'], _frame0.mention.get_words())
                    self.check_string_match(rr['text_arg1'], _frame1.mention.get_words())
                    c0, c1 = _frame0.sent.get_text(), _frame1.sent.get_text()
                    self.check_string_match(rr['text_context'], c0 if c0==c1 else f"{c0}\n{c1}")
        # --
        if output_path:
            with WriterGetterConf().get_writer(output_path=output_path) as writer:
                writer.write_insts(all_insts)
        # --
        return cc

    # write/update data
    # update_option if existing: skip, update
    def run_update(self, col, input_path: str, ff='', update_option='update'):
        conf: MdbHelperConf = self.conf
        cc = Counter()
        ff = self.get_f(ff, 'x')
        # --
        col_doc, col_ent, col_rel = [self.get_cols(col+z, get_one=True) for z in ["", conf.col_ent, conf.col_rel]]
        reader = ReaderGetterConf().get_reader(input_path=input_path)
        for inst in reader:
            if ff and not ff(inst):
                continue
            cc['doc'] += 1
            # --
            _doc_dd = {'doc_id': inst.id, 'text': inst.get_text()}
            if inst.info:
                _doc_dd['info'] = inst.info
            if inst.sents:
                _doc_dd['sents'] = [s.seq_word.vals for s in inst.sents]
            assert inst.id is not None
            existing_docs = list(col_doc.find({'doc_id': inst.id}))
            if len(existing_docs) == 0:  # new one, simply add
                cc['doc_new'] += 1
                col_doc.insert_one(_doc_dd)
                assert col_ent.count_documents({'doc_id': inst.id}) == 0
                assert col_rel.count_documents({'doc_id': inst.id}) == 0
            else:
                assert len(existing_docs) == 1
                existing_doc = existing_docs[0]
                if update_option == 'skip':
                    cc['doc_skip'] += 1
                    continue
                assert existing_doc['text'] == _doc_dd['text']
                col_doc.find_one_and_replace({'doc_id': inst.id}, _doc_dd)
                # delete existing anns
                res0, res1 = col_ent.delete_many({'doc_id': inst.id}), col_rel.delete_many({'doc_id': inst.id})
                cc['frame_del'] += res0.deleted_count
                cc['alink_del'] += res1.deleted_count
            # add new anns
            _fs, _as = self._export_anns(inst)
            cc['frame_new'] += len(_fs)
            cc['alink_new'] += len(_as)
            if _fs:
                col_ent.insert_many(_fs)
            if _as:
                col_rel.insert_many(_as)
        return cc

    # helper for inserting frames
    def _export_anns(self, doc):
        # --
        def _mention2text(_m):
            _sent = _m.sent
            if _sent.word_positions is None:
                return _m.get_words(concat=True)  # simply concat
            else:  # get original
                _widx, _wlen = _m.get_span()
                _wridx = _widx + _wlen - 1
                _start, _end = _sent.word_positions[_widx][0], sum(_sent.word_positions[_wridx])
                return _sent.doc.get_text()[_start:_end]
        # --
        # simply insert all!
        doc_id = doc.id
        all_frames = OrderedDict()
        for frame in doc.get_frames():
            dd = {'doc_id': doc_id, 'frame_id': f"{frame.cate}:{frame.id}",
                  'position': (frame.mention.sid, ) + frame.mention.get_span(), 'label': frame.label}
            if frame.info:
                dd['info'] = frame.info
            dd['text'] = _mention2text(frame.mention)
            dd['text_context'] = frame.sent.get_text()
            assert dd['frame_id'] not in all_frames
            all_frames[dd['frame_id']] = dd
        all_rels = []
        for frame in doc.get_frames():
            for alink in frame.args:
                arg0, arg1 = f"{alink.main.cate}:{alink.main.id}", f"{alink.arg.cate}:{alink.arg.id}"
                assert arg0 in all_frames and arg1 in all_frames
                dd = {'doc_id': doc_id, 'arg0': arg0, 'arg1': arg1, 'label': alink.label}
                if alink.info:
                    dd['info'] = alink.info
                dd.update({'text_arg0': all_frames[arg0]['text'], 'text_arg1': all_frames[arg1]['text']})
                c0, c1 = all_frames[arg0]['text_context'], all_frames[arg1]['text_context']
                dd['text_context'] = c0 if c0==c1 else f"{c0}\n{c1}"
                all_rels.append(dd)
        return list(all_frames.values()), all_rels

# --
def main(*args):
    conf: MainConf = init_everything(MainConf(), args)  # real init!
    helper = MdbHelper(conf.mdb)
    helper.run(conf.cmd)
    # --

# PYTHONPATH=../src/ python3 -m mspx.tools.al.utils_mdb ...
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
# --

# =====
# ## setup ##
# install
INSTALL = """
# https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu bionic/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
# echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org
"""
START = """
# sudo systemctl start mongod
# auth
# https://www.mongodb.com/docs/manual/tutorial/configure-scram-client-authentication/
mongod -f mongod.conf
mongosh
use admin
db.createUser(
  {
    user: "myUserAdmin",
    pwd: passwordPrompt(), // or cleartext password
    roles: [
      { role: "userAdminAnyDatabase", db: "admin" },
      { role: "readWriteAnyDatabase", db: "admin" }
    ]
  }
)
# https://www.mongodb.com/docs/manual/tutorial/create-users/
mongod -f mongod.conf --auth
mongosh -u "myUserAdmin" -p
use nlp_mat
db.createUser(
  {
    user: "mat",
    pwd:  passwordPrompt(),   // or cleartext password
    roles: [ { role: "readWrite", db: "nlp_mat" } ]
  }
)
# --
mongosh "mongodb://IP" -u "myUserAdmin" -p
mongosh "mongodb://IP" --authenticationDatabase "nlp_mat" -u "mat" -p
"""
PYMONGO = """
pip install pymongo
"""
def do_test():
    from pymongo import MongoClient
    uri = "mongodb://%s:%s@%s/%s" % ('mat', '***', 'IP', 'nlp_mat')
    client = MongoClient(uri, authSource="nlp_mat")
    db = client.nlp_mat
    db.list_collection_names()
    # --
    posts = db.posts
    post = {"author": "Mike", "text": "My first blog post!", "tags": ["mongodb", "python", "pymongo"]}
    p = posts.insert_one(post)
    # --
# =====
# run
"""
python3 -m mspx.tools.al.utils_mdb "uri:mongodb://mat:**@localhost:27018/nlp_mat" "cmd:list data_queue"
# ->
"cmd:update ace ../w22/data/evt/data/en.ace05.dev.json"
"cmd:read ace tmp.json"
"cmd:qutils"
"""
# example queries
"""
[{$group: {_id: "$label", count: {$count: {}}}}, {$sort: {count: -1}}]
[{$match: {label: "Result"}}, {$group: {_id: "$text", count: {$count: {}}}}, {$sort: {count: -1}}, {$match: {count: {$gte: 2}}}]
"""
