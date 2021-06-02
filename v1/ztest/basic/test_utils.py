#

from msp import utils
from msp.utils import zlog, zcheck, StatRecorder, Random, Helper, Conf

class Conf0(Conf):
    def __init__(self):
        self.x = 1
        self.y = "test"
        self.z = Conf1()

class Conf1(Conf):
    def __init__(self):
        self.x = "k"
        self.z = "what"
        self.a = 100

def main():
    utils.init("zlog", 1234)
    z = StatRecorder(True)
    times = Random.randint(100)
    for _ in range(times):
        with z.go():
            z.record_kv("distr_n", Random.randint(10))
    Helper.printd(z.summary(), "\n")
    #
    cc = Conf0()
    cc.update_from_args(["a:10", "y:www", "z.x:1"])
    pass

if __name__ == '__main__':
    main()
