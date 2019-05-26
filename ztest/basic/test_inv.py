#

# using LU to get inv and det

import torch

def main():
    for BS in [1, 10, 100]:
        for L in [4, 16, 51]:
            x = torch.randn([BS, L, L])
            diag1 = torch.eye(L)
            x_inv, x_lu = diag1.gesv(x)
            x_det = (x_lu * diag1).sum(-1).prod(-1)
            # check
            for idx in range(BS):
                cur_inv = torch.inverse(x[idx])
                cur_det = torch.det(x[idx])
                calc_inv = x_inv[idx]
                calc_det = x_det[idx]
                #
                THR = 1e-2
                assert float(torch.sum(torch.abs(cur_inv - calc_inv))) / (L*L) < THR
                # todo(warn): lack permutation P, whose det can be -1 or 1
                assert abs(float(cur_det))-abs(float(calc_det)) < THR
            print("OK with %s and %s." % (BS, L))

if __name__ == '__main__':
    main()
