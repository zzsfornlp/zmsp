#

# todo-list
# topical influence (use actual bert's predicted output to delete special semantics?) -> change words: first fix non-changed, then find repeated topic words, then change topic and hard words one per segment -> (191103: change too much may hurt)
# predict-leaf (use first half of order as leaf, modify the scores according to this first half?) -> (191104: still not good)
# other reduce criterion? -> (191105: not too much diff, do not get it too complex...)
# vocab based reduce? simply <= thresh? -> (191106: ok, help a little, but not our target)
# cky + decide direction later
# only change topical words?
# direct parse subword seq? (group and take max/avg-score)
# cluster (hierachical bag of words, direction?, influence range, grouped influence)
