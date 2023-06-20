#

# a script to test gpu

import sys
import torch
import time

def main(sleep=0.):
    sleep = float(sleep)
    BS = 100
    M = 1000
    x1 = torch.rand([BS, M, M]).cuda()
    x2 = torch.rand([BS, M, M]).cuda()
    while True:
        y = torch.matmul(x1, x2)
        if sleep > 0:
            time.sleep(float(sleep))
    # --

if __name__ == '__main__':
    main(*sys.argv[1:])
