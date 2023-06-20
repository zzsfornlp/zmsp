#

# simply for testing

import pandas
from mspx.znew.models.model_api import *

def main():
    m = NmApiConf().make_node()
    # m = NmApiConf().direct_conf(model_name='text-ada-001').make_node()
    m.run_ppl(["What is the nature of life? That is ", "What is the meaning of life? That is "], ["repeater", "nonsense"])

# --
# simple test
# python3 -m mspx.cli.ztest
if __name__ == '__main__':
    main()
# --
