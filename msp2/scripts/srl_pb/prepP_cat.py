#

# simply concatenate for PB data

import os

# --
PB_DIR="propbank-release"

# --
def main():
    for dname in ["ontonotes", "ewt"]:
        for wset in ["train", "dev", "test", "conll12-test"]:
            if dname=="ewt":
                if wset=="conll12-test": continue
                fname = f"{dname}.{wset}.txt"
                data_prefix = f"{PB_DIR}/data/google/ewt/"
            else:
                fname = f"{dname}-{wset}-list.txt"
                data_prefix = f"{PB_DIR}/data/"
            # read all files and write
            with open(f"{dname}.{wset}.conll", 'w') as wfd:
                cur_hit = 0
                cur_miss = 0
                with open(f"{PB_DIR}/docs/evaluation/{fname}") as rfd:
                    for line in rfd:
                        one_name = line.rstrip()
                        if one_name.endswith(".conllu"):
                            one_name = ".".join(one_name.split(".")[:-1])
                        cur_data_file = f"{data_prefix}/{one_name}.gold_conll"
                        if not os.path.isfile(cur_data_file):
                            cur_miss += 1
                            # note: misses are mainly that currently PB does not include sentences without predicates
                            print(f"Miss file: {cur_data_file}")
                        else:
                            cur_hit += 1
                            with open(f"{data_prefix}/{one_name}.gold_conll") as rfd2:
                                wfd.write(rfd2.read())
                # --
                print(f"Write {wfd} with {cur_hit}/{cur_miss} files.")
    # --

# python3 prepP_cat.py
if __name__ == '__main__':
    main()
