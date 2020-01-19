#

# ready go!

"""
TODO(+N):
- currently cost does not count future cost (by acyclic constraint), which may bring troubles?
- currently still not full dynamic oracle
-----
- 1) weight by cost for the final loss,
- 1.5) check other systems (like td/nf/lr)
- 1.6) check speed
- 2) other enders, like early-update/max-violation/BSO
- 2.5) refining the parts about the losses
-- (arc/label: cost(arc_wrong=>label-wrong), oracle_states(what to include), separate arc/label & with diff div)
- 2.6) features (par/chs)
# =====
- ??) multi-margin?
- 3) high order graph and init with graph trained parser
TODO(+N):
    4) acording to profiling, most of the time are still on State-building, which can be surely speed up
"""
