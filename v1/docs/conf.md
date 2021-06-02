## How to configurate the running (cmd) options:

The specifications of the configurations are based on the `Conf` class in [`msp/utils/conf.py`](../msp/utils/conf.py). The to-be-provided cmd options are slightly different than the conventional ones in C++ programs or those in argparse, but may also be straight-forward.

Each token (usually as splitted by the shell and provided to the program) in the `sys.argv` specify one entry of configuration. Each token is a key-value pair with `:` as the separator between key and value. (Notice that there are no prefixes of `--` and no spaces between the key and value.) From system arguments, the values are initially all `str`, a cast to the entry's type is applied when assigning values.

For example, in the GraphParser example, that `GraphParserConf` ([`tasks/zdpar/graph/parser.py`](../tasks/zdpar/graph/parser.py)) class has an attribute named `output_normalizing` which is initialized as a string. Then a cmd argument of `output_normalizing:global` will change that field to `global`.

The `Conf` class can be composed hierarchically, for example, the `GraphParserConf` has a `sc_conf` field with the type of `GraphScorerConf`. Then, the specification of fields in that class can also be hierarchical, for example, `sc_conf.arc_space:1024` will set `sc_conf.arc_space` to `1024`. For a full example, please check the running scripts of the GraphParser.

----

Here is a toy example:

~~~~python
# test_conf.py
import sys
from msp.utils import Conf

class Conf0(Conf):
    def __init__(self):
        self.c1 = Conf1()
        self.c2 = Conf2()

class Conf1(Conf):
    def __init__(self):
        self.x = "k"
        self.y = "what"
        self.z = 100

class Conf2(Conf):
    def __init__(self):
        self.x = 1.2
        self.y = ""
        self.z2 = 100

if __name__ == '__main__':
	cc = Conf0()
	cc.update_from_args(sys.argv[1:])

# python3 test_conf.py "c1.x:abc" "c2.x:10." "z2:101" "c2.z2:102"
~~~~
