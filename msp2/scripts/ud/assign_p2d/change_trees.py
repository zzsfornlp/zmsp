#

# gather trees (with node drops)

import sys

# --
# drop tree nodes
TAG_SCHEME = {
    # drop EDITED for en
    "en": {"EDITED": None},
}
# change word chars
CHAR_SCHEME = {
    "en": {'\\': "<"},
    "zh": {'＜': "<", '＞': ">", '［': "<", '］': ">", '{': "<", '}': ">", '｛': "<", '｝': ">", '〈': "<", '〉': ">"},
}
# --

# --
class TreeNode:
    def __init__(self, tag: str, word: str = None):
        self.tag = tag
        self.word = word
        self.chs = []

    def add_ch(self, ch: 'TreeNode'):
        assert not self.is_leaf()
        self.chs.append(ch)

    def is_leaf(self):
        ret = (self.word is not None)
        if ret:
            assert len(self.chs) == 0
        return ret

    # change them all!
    def to_string(self, tag_map: dict, char_map: dict):
        _tag = self.tag
        cur_tag = tag_map.get(_tag, _tag)
        if cur_tag is None:
            return ""  # deleted!
        if self.is_leaf():
            str_word = ''.join([char_map.get(c, c) for c in self.word])
            return f"({cur_tag} {str_word})"
        else:
            str_chs = " ".join([ch.to_string(tag_map, char_map) for ch in self.chs])
            if len(str_chs)==0 or str.isspace(str_chs): return ""  # also remove propagated empty ones!
            return f"({cur_tag} {str_chs})"
        # --

class TreeReader:
    def __init__(self):
        pass

    def _yield_char_from_fd(self, fd):
        for line in fd:
            yield from line
        # --

    # tokenize: input stream of chars
    def _tokenize(self, stream):
        cur = []
        for c in stream:
            is_space = str.isspace(c)
            is_bracket = (c in "()")
            if is_space or is_bracket:  # end of a token
                if len(cur) > 0:
                    yield ''.join(cur)
                    cur.clear()
            if is_space:
                continue
            elif is_bracket:
                yield c
            else:
                cur.append(c)
            # --
        if len(cur) > 0:
            yield ''.join(cur)
        # --

    # parse: input stream of tokens
    def _parse(self, stream):
        stack = []
        for tok in stream:
            if tok == '(':  # open one
                node = TreeNode(None)
                stack.append(node)
            elif tok == ')':
                ch = stack.pop()
                assert ch.is_leaf() or len(ch.chs)>0
                if len(stack) == 0:
                    yield ch  # already top level
                else:  # add ch
                    stack[-1].add_ch(ch)
            else:
                node = stack[-1]
                if node.tag is None:
                    node.tag = tok
                else:  # leaf
                    assert len(node.chs) == 0
                    node.word = tok
        assert len(stack) == 0
        # --

    def yield_from_fd(self, fd):
        char_stream = self._yield_char_from_fd(fd)
        tok_stream = self._tokenize(char_stream)
        parse_stream = self._parse(tok_stream)
        yield from parse_stream
# --

def main(scheme: str):
    tag_map = TAG_SCHEME.get(scheme, {})  # by default change nothing
    char_map = CHAR_SCHEME.get(scheme, {})  # by default change nothing
    # --
    reader = TreeReader()
    for tree in reader.yield_from_fd(sys.stdin):
        ss = tree.to_string(tag_map, char_map)
        sys.stdout.write(ss+"\n")
    # --

if __name__ == '__main__':
    main(*sys.argv[1:])
    # --

# python3 change_trees.py [scheme] <? >?
