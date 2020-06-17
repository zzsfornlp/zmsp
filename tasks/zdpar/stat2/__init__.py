#

# trying-list
# topical influence (use actual bert's predicted output to delete special semantics?) -> change words: first fix non-changed, then find repeated topic words, then change topic and hard words one per segment -> (191103: change too much may hurt)
# predict-leaf (use first half of order as leaf, modify the scores according to this first half?) -> (191104: still not good)
# other reduce criterion? -> (191105: not too much diff, do not get it too complex...)
# vocab based reduce? simply <= thresh? -> (191106: ok, help a little, but not our target)
# cky + decide direction later -> (191111: still not good, but first pdb-debug; 191112: still not good)
# cky + two-layer by splitting puncts? -> (191113: only slightly helpful)
# stop words (<100 in voacb) as lower ones? -> (191114: worse than pos-rule)
# check wsj and phrase result? -> (191114: f1 around 40)
# only change topical words? -> (191115: change 883/25148, no obvious diff)
# direct parse subword seq? (group and take max/avg-score) -> (191115: max-score seems to be slightly helpful +1)
# cluster (hierachical bag of words, direction?, influence range, grouped influence) -> (191115: similar, but slightly worse than cky)
# -- ok, here is the end, goto next stage ...
