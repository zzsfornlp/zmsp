#

# specific conllu reader for UD conllu files

from msp.utils import zcheck, FileHelper

# one instance of parse
class ConlluToken:
    def __init__(self, parse, idx, w, up, xp, h, la, enh_hs, enh_ls, misc=None):
        self.parse = parse
        # basic properties (on construction)
        self.idx = idx
        self.word = w
        self.upos = up
        self.xpos = xp
        self.head = h
        self.label = la
        self.label0 = la.split(":")[0] if la is not None else None
        self.misc = {}
        if misc is not None and len(misc)>0 and misc != "_":
            try:
                for s in misc.split("|"):
                    k, v = s.split("=")
                    assert k not in self.misc, f"Err: Repeated key: {k}"
                    self.misc[k] = v
            except:
                pass
        # enhanced dependencies
        self.enh_heads = enh_hs
        self.enh_labels = enh_ls
        #
        # extra helpful ones (build later, todo(warn) but not for ellip nodes!!)
        # dep-distance and root-distance
        self.ddist = 0      # mod - head
        self.rdist = -1     # distance to root
        # self.rdist = 0
        # left & right & all child (sorted left-to-right)
        self.lchilds = []
        self.lchilds_labels = []
        self.rchilds = []
        self.rchilds_labels = []
        self.childs = []
        self.childs_labels = []

    def __repr__(self):
        return "\t".join(str(z) for z in (self.idx, self.word, self.upos, self.xpos, self.head, self.label0, self.misc))

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            misc = self.__dict__.get("misc", None)
            if misc is not None and item in misc:
                return misc[item]
            else:
                raise AttributeError()

    def is_root(self):
        return self.idx == 0

    def is_leaf(self):
        return len(self.childs)==0

    def is_ellip(self):
        return not isinstance(self.idx, int)

    def has_enhance(self):
        return self.enh_heads is not None

    def has_multihead(self):
        return self.has_enhance() and len(self.enh_heads)>0

    # get the node by idx from the same tree
    def get_node(self, idx):
        return self.parse.get_tok(idx)

    def get_head(self):
        return None if (self.head is None or self.head<0) else self.get_node(self.head)

class ConlluParse:
    ROOT_SYM = "<R>"
    NUM_FIELDS = 10
    SKIP_F = lambda x: len(x.strip()) == 0
    IGNORE_F = lambda x: x.startswith("#")

    def __init__(self):
        # headlines
        self.headlines = []
        # ordinary tokens
        self.tokens = []        # [0] is root token
        # enhance tokens: {int/tuple[id,n]: *}
        self.enhance_tokens = {}
        # crossed?
        self._crossed = None
        # other properties
        self.misc = {}

    @staticmethod
    def calc_crossed(heads):
        # todo(warn): does not include ROOT
        cr = [0] * len(heads)
        # todo(warn): naive one
        # check each pair of hm-pair
        pairs = [(m, h) for m, h in enumerate(heads, 1)]
        for m0, h0 in pairs:
            if cr[m0 - 1]:
                continue
            for m1, h1 in pairs:
                if m1 != m0:
                    min0, max0 = min(m0, h0), max(m0, h0)
                    min1, max1 = min(m1, h1), max(m1, h1)
                    if min1 < min0 and min0 < max1 and max1 < max0:
                        cr[m0 - 1] = 1
                        cr[m1 - 1] = 1
                    if min0 < min1 and min1 < max0 and max0 < max1:
                        cr[m0 - 1] = 1
                        cr[m1 - 1] = 1
        return cr

    @property
    def crossed(self):
        # compute on need
        if self._crossed is None:
            cr = ConlluParse.calc_crossed([z.head for z in self.get_tokens()])
            self._crossed = cr
        return self._crossed

    def __repr__(self):
        return "\n".join([str(z) for z in self.tokens])

    # add comment
    def add_headline(self, s):
        self.headlines.append(s)

    # sequentially add one
    def add_token(self, k, t: ConlluToken, keep_seq=True):
        if isinstance(k, int):
            if keep_seq:
                zcheck(k == len(self.tokens), "Conllu Error: Not added in sequence!")
                self.tokens.append(t)
            else:
                self.tokens[k] = t
        else:
            zcheck(isinstance(k, tuple) and len(k)==2, "Conllu Error: Illegal ellipsis key")
        self.enhance_tokens[k] = t

    # after collecting all tokens, build more
    def finish_tokens(self):
        # put childs (left to right)
        for cur_idx in range(1, len(self.tokens)):
            cur_tok = self.tokens[cur_idx]
            cur_label = cur_tok.label
            head_idx = cur_tok.head
            zcheck(cur_idx != head_idx, "Conllu Error: no self-link.")
            head_tok = self.tokens[head_idx]
            cur_ddist = cur_idx - head_idx
            if cur_ddist > 0:       # mod after head
                head_tok.rchilds.append(cur_idx)
                head_tok.rchilds_labels.append(cur_label)
            else:
                head_tok.lchilds.append(cur_idx)
                head_tok.lchilds_labels.append(cur_label)
            head_tok.childs.append(cur_idx)
            head_tok.childs_labels.append(cur_label)
            cur_tok.ddist = cur_ddist
        # calculate root distance
        # =====
        def _calc_rdist(tok, d):
            tok.rdist = d
            for ii in tok.childs:
                _calc_rdist(self.tokens[ii], d+1)
        # =====
        _calc_rdist(self.tokens[0], 0)

    # get real tokens (exclude root)
    def get_tokens(self):
        return self.tokens[1:]

    # idx=0 is the root
    def get_tok(self, idx):
        return self.tokens[idx]

    #
    def get_props(self, name):
        return [getattr(z, name) for z in self.tokens[1:]]

    def __len__(self):
        return len(self.tokens)-1

    # yield (tok, [parents])
    # todo(warn): by default returned stack is being modified
    def iter_depth_first(self, copy_stack=False):
        def _iter_depth_first(cur_stack):
            cur_node = cur_stack[-1]
            if copy_stack:
                yield (cur_node, cur_stack.copy())
            else:
                yield (cur_node, cur_stack)
            #
            for idx in cur_node.childs:
                child_node = self.tokens[idx]
                cur_stack.append(child_node)
                for rone in _iter_depth_first(cur_stack):
                    yield rone
                cur_stack.pop()
        #
        cur_stack = [self.tokens[0]]
        for one in _iter_depth_first(cur_stack):
            yield one

    # todo(warn): only outputting upos
    def write(self, fd):
        write_conllu(fd, self.get_props("word"), self.get_props("upos"), self.get_props("head"), self.get_props("label"))

class ConlluReader:
    # parse the idx, return int or (N,M)
    def parse_idx(self, id_str):
        split_elli = id_str.split(".")
        if len(split_elli)>1:
            zcheck(len(split_elli)==2, "Conllu Error: strange ELLI line")
            one_idx = (int(split_elli[0]), int(split_elli[1]))
        else:
            one_idx = int(id_str)
        return one_idx

    # get enhanced dependency graph, return (enh-heads, enh-labels)
    def parse_enhance_dep(self, enh_str):
        if enh_str == "_":
            return None, None
        else:
            hs, ls = [], []
            for one_dep in enh_str.split("|"):
                one_h, one_l = one_dep.split(":", 1)
                one_h = self.parse_idx(one_h)
                hs.append(one_h)
                ls.append(one_l)
            return hs, ls

    def read_one(self, fd):
        # lines = FileHelper.read_multiline(fd, ConlluParse.SKIP_F, ConlluParse.IGNORE_F)
        lines = FileHelper.read_multiline(fd, ConlluParse.SKIP_F)
        if lines is None:
            return None
        #
        RR = ConlluParse.ROOT_SYM
        parse = ConlluParse()
        root_token = ConlluToken(parse, 0, RR, RR, RR, -1, None, None, None, None)
        parse.add_token(0, root_token)
        #
        prev_idx = 0
        for one_line in lines:
            if one_line.startswith("#"):
                # add headlines
                parse.add_headline(one_line[1:].strip())
                continue
            fileds = one_line.strip("\n").split('\t')
            zcheck(len(fileds)==ConlluParse.NUM_FIELDS, "Conllu Error: Unmatched num of fields.")
            # id
            id_str = fileds[0]
            split_mwe = id_str.split("-")
            if len(split_mwe)>1:
                # currently, not using these lines
                zcheck(len(split_mwe)==2, "Conllu Error: strange MWE line")
                continue
            one_idx = self.parse_idx(id_str)
            if isinstance(one_idx, int):
                prev_idx += 1
                zcheck(one_idx == prev_idx, "Conllu Error: wrong line-idx")
            else:
                zcheck(one_idx[0] == prev_idx, "Conllu Error: wrong ELLI line-idx")
            # add basic ones
            word, upos, xpos, head, label, misc = fileds[1], fileds[3], fileds[4], fileds[6], fileds[7], fileds[9]
            head = int(head) if head!="_" else None
            ehn_hs, enh_ls = self.parse_enhance_dep(fileds[8])
            one_token = ConlluToken(parse, one_idx, word, upos, xpos, head, label, ehn_hs, enh_ls, misc)
            parse.add_token(one_idx, one_token)
        parse.finish_tokens()
        return parse

    def yield_ones(self, fd):
        while True:
            parse = self.read_one(fd)
            if parse is None:
                break
            yield parse

# writer
def write_conllu(fd, word, pos, heads, labels, miscs=None, headlines=None):
    length = len(word)
    if miscs is None:
        miscs = ["_"] * length
    if headlines is not None:
        for line in headlines:
            fd.write(f"# {line}\n")
    for idx in range(length):
        fields = ["_"] * 10
        fields[0] = str(idx+1)
        fields[1] = word[idx]
        fields[3] = pos[idx]
        fields[6] = str(heads[idx])
        fields[7] = labels[idx]
        if len(miscs[idx])>0:
            fields[9] = miscs[idx]
        s = "\t".join(fields) + "\n"
        fd.write(s)
    fd.write("\n")
