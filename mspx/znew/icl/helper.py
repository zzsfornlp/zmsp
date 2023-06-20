#

# some helper functions

def len_spath(x, upper=4):
    return min(len(x['spath']), upper)

def sel_spath(x):
    x = '_'.join(x['spath'])
    if x in 'nsubj,nsubj:pass,nmod,obj,obl,compound'.split(","):
        return x
    else:
        return "others"

def score_spath(x, y):
    from mspx.tools.algo.align import AlignHelper
    s1, s2 = x['spath'], y['spath']
    dist = AlignHelper.edit_distance(s1, s2)
    ndist = dist / max(len(s1), len(s2))
    return 1. - ndist
