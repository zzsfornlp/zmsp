#
import sys
# convert between several forms of tab files & also calculate num-s/w

# return [[f1,f2,f3,...], ...]
def read_one(fd, iform):
    # separated by multiple "\n"
    words = None
    while True:
        line = fd.readline()
        if len(line) <= 0:
            return None
        words = line.strip().split("\t")
        # some ud files begins with #
        # todo(warn): specific judgement for comment line
        if len(words) > 0 and len(words[0])>0 and not (words[0].startswith("#") and len(words) <= 1):
            break
    rets = []
    while len(words) > 0:
        try:
            # ud file with "1-2" as id
            if "ID" in iform[1]:
                idx = int(words[iform[1]["ID"]])
            rets.append(words)
        except:
            pass
        line = fd.readline()
        words = line.strip().split("\t")
        if len(line) <= 0 or len(words) <= 0 or len(words[0]) <= 0:
            break
    return rets

def read(f, iform):
    them = []
    with open(f) as fd:
        while True:
            one = read_one(fd, iform)
            if one is None:
                break
            them.append(one)
    print("Reading from %s, %s" % (f, stat(them)))
    return them

def write(f, them):
    print("Writing to %s, %s" % (f, stat(them)))
    with open(f, "w") as fd:
        for one in them:
            for fs in one:
                fd.write("\t".join(fs)+"\n")
            fd.write("\n")

def stat(them):
    num_sents = len(them)
    num_words = sum(len(i) for i in them)
    return num_sents, num_words

def stat_proj(them, tt):
    # naive calculation
    num_sents = 0
    num_words = 0
    num_sents_proj = 0
    num_words_proj = 0
    head_idx = tt[1]["HEAD"]
    for sent in them:
        num_sents += 1
        proj_sent = True
        spans = ((i+1, int(word[head_idx])) for i, word in enumerate(sent))
        for i, word in enumerate(sent):
            num_words += 1
            m, h = i+1, int(word[head_idx])
            proj_word = True
            for a,b in spans:
                l,r = min(a,b), max(a,b)
                l2, r2 = min(m,h), max(m,h)
                if (l<l2<r and r<r2) or (l2<l<r2 and r2<r):
                    proj_sent = False
                    proj_word = False
                    break
            if proj_word:
                num_words_proj += 1
        if proj_sent:
            num_sents_proj += 1
    results = (num_sents, num_sents_proj, num_sents_proj/num_sents, num_words, num_words_proj, num_words_proj/num_words)
    print("Sent=(%s,%s,%.3f), Words=(%s,%s,%.3f)" % results)
    print("results: %d %d %.3f %d %d %.3f" % results)

# converting or combining
# ft/to is (num, dicts) for key names
def convert(them, fr, to):
    ret = []
    for one in them:
        item = []
        for i, fs in enumerate(one):
            assert len(fs) == fr[0]
            fs2 = ["_" for ii in range(to[0])]
            for k in to[1]:
                if k == "ID":
                    v = str(i+1)
                else:
                    v = fs[fr[1][k]]
                fs2[to[1][k]] = v
            item.append(fs2)
        ret.append(item)
    return ret

def add(base, bt, adder, at):
    right, wrong = 0, 0
    assert len(base) == len(adder)
    # can only change one field
    change_field = None
    for bb, aa in zip(base, adder):
        assert len(bb) == len(aa)
        for bone, aone in zip(bb, aa):
            # init
            if change_field is None:
                for k in at[1]:
                    if aone[at[1][k]] != bone[bt[1][k]]:
                        change_field = k
                        print("Adding one field of %s." % change_field)
                        break
            changed = False
            for k in at[1]:
                if k == change_field:
                    if bone[bt[1][k]] != aone[at[1][k]]:
                        bone[bt[1][k]] = aone[at[1][k]]
                        changed = True
                else:
                    assert bone[bt[1][k]] == aone[at[1][k]]
            if changed:
                wrong += 1
            else:
                right += 1
    print("Adding rate is %s/%s/%s/%.3f." % (right, wrong, right+wrong, 100.*right/(right+wrong)))
    return base

TABLES = {
    "conll06": (10, {"ID":0, "FORM":1, "POS":3, "HEAD":6, "LABEL":7}),
    "conll08": (10, {"ID":0, "FORM":1, "POS":3, "HEAD":8, "LABEL":9}),
    "conllu": (10, {"ID":0, "FORM":1, "UPOS":3, "POS":4, "HEAD":6, "LABEL":7}),
    "tab": (4, {"FORM":0, "POS":1, "HEAD":2, "LABEL":3}),
    "pos": (2, {"FORM":0, "POS":1}),
    "form": (1, {"FORM":0})
}

def main():
    # cmd is: python conll.py <input-file> <input-form> <output-file> <output-form> (opt <adder-file> <adder-form>)
    ifile, iform = sys.argv[1], TABLES[sys.argv[2]]
    ithem = read(ifile, iform)
    if len(sys.argv) > 3:
        ofile, oform = sys.argv[3], TABLES[sys.argv[4]]
        othem = convert(ithem, iform, oform)
        if len(sys.argv) > 5:
            afile, aform = sys.argv[5], TABLES[sys.argv[6]]
            athem = read(afile, aform)
            othem = add(othem, oform, athem, aform)
        write(ofile, othem)
    else:
        # telling stat
        stat_proj(ithem, iform)

if __name__ == '__main__':
    main()

# from table
import re, numpy

def draw():
    all_results = []
    while True:
        try:
            line = input()
        except:
            break
        if len(line) == 0:
            break
        res = re.findall('([0-9][0-9]\.[0-9][0-9])', line)
        if len(res) > 0:
            all_results.append([float(r) for r in res])
    them = numpy.asarray(all_results)
    print(them)
    print(them.T)
    print(numpy.average(them, axis=0))
    #
    print()
    for ones in all_results:
        print("\t".join([str(r) for r in ones]))
