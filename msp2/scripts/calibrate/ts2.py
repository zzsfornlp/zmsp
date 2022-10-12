#

# another version of ts

# =====
# note: specific data reader!!
def read_data(file: str):
    import numpy as np
    from msp2.data.inst import yield_frames, set_ee_heads, DataPadder
    from msp2.data.rw import ReaderGetterConf
    from msp2.tasks.zmtl3.mod.extract.evt_arg import onto as zonto
    from collections import Counter
    cc = Counter()
    # --
    onto = zonto.Onto.load_onto("pbfn")
    reader = ReaderGetterConf().get_reader(input_path=file)
    all_logits = []
    utilized_roles = {"ARGM-LOC", "ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"}
    for inst in reader:
        set_ee_heads(inst)
        for frame in yield_frames(inst):
            cc['frame0'] += 1
            if onto.find_frame(frame.label) is None:
                continue
            for arg in frame.args:
                cc['arg0'] += 1
                gold_id = arg.mention.shead_token.get_indoc_id(True)
                _role = arg.role
                if _role in utilized_roles:
                    cc['arg1'] += 1
                    vv = frame.info['arg_scores'][_role]
                    if gold_id not in vv:
                        print("Gold token out of reach!!")
                        continue
                    cc['arg2'] += 1
                    one_logits = [vv[gold_id]] + [v for k,v in vv.items() if k!=gold_id] + [0]  # add 0!
                    all_logits.append(one_logits)
    # --
    # finally concat all
    ret_logits = DataPadder.go_batch_2d(all_logits, pad_val=-100.)[:,:,None]
    ret_labels = np.asarray([0] * len(ret_logits))
    print(f"Load from {file} ({ret_logits.shape}, {ret_labels.shape}): {cc}")
    return ret_logits, ret_labels
# =====

# --
def main(input_file: str, optimizer='adam', device=0):
    from msp2.scripts.calibrate.ts import set_temperature, Adam_optimizer, LBFGS_optimizer
    # --
    n_epochs = 10000
    if optimizer == 'adam':
        optimizer = Adam_optimizer(n_epochs)
    elif optimizer == 'lbfgs':
        optimizer = LBFGS_optimizer(n_epochs)
    else:
        print("Illegal optimizer {}, must be one of (adam, lbfgs)")
        return -2
    # --
    rets = set_temperature(read_data(input_file), 1, optimizer, int(device))
    print(f"Final values are: {rets}")
    # --

if __name__ == '__main__':
    import sys
    sys.exit(main(*sys.argv[1:]))

# --
"""
python3 -m msp2.scripts.calibrate.ts2 ../qadistill/en.ewt.dev.ud.{q1,q2}.json {adam,lbfgs}
# q1 -> Final values are: [2.473419189453125]/[2.649324893951416]
# q2 -> Final values are: [2.0260472297668457]/[2.1015663146972656]
"""
