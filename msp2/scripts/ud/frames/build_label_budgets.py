#

#

# from span-srl to dep-srl with dep tree

from collections import Counter
from msp2.data.inst import yield_sents
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.utils import zlog, default_json_serializer

# --
def main(input_file: str, max_budget=1, output_file=''):
    cc = Counter()
    max_budget = int(max_budget)
    all_insts = list(ReaderGetterConf().get_reader(input_path=input_file))
    from msp2.data.vocab.frames import RoleBudgetHelper
    budgets = RoleBudgetHelper.build_role_budgets_from_data(all_insts, max_budget=max_budget)
    if output_file in ["", "-"]:
        # zlog(budgets)
        # better printing
        for k in sorted(budgets.keys()):
            zlog(f"'{k}': {budgets[k]},")
    else:
        default_json_serializer.to_file(budgets, output_file)
    # --

# --
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

# --
# PYTHONPATH=?? python3 build_label_budgets.py IN OUT
