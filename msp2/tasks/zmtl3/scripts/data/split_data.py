# --

import os
import json

def split_data(input_file, output_dir, split_path):
    """Splits the input file into train/dev/test sets.

    Args:
        input_file (str): path to the input file.
        output_dir (str): path to the output directory.
        split_path (str): path to the split directory that contains three files,
            train.doc.txt, dev.doc.txt, and test.doc.txt . Each line in these
            files is a document ID.
    """
    print('Splitting the dataset into train/dev/test sets')
    train_docs, dev_docs, test_docs = set(), set(), set()
    # Load doc ids
    with open(os.path.join(split_path, 'train.doc.txt')) as r:
        train_docs.update(r.read().strip('\n').split('\n'))
    with open(os.path.join(split_path, 'dev.doc.txt')) as r:
        dev_docs.update(r.read().strip('\n').split('\n'))
    with open(os.path.join(split_path, 'test.doc.txt')) as r:
        test_docs.update(r.read().strip('\n').split('\n'))

    # Split the dataset
    with open(input_file, 'r', encoding='utf-8') as r, \
            open(os.path.join(output_dir, 'train.oneie.json'), 'w') as w_train, \
            open(os.path.join(output_dir, 'dev.oneie.json'), 'w') as w_dev, \
            open(os.path.join(output_dir, 'test.oneie.json'), 'w') as w_test:
        for line in r:
            inst = json.loads(line)
            doc_id = inst['doc_id']
            if doc_id in train_docs:
                w_train.write(line)
            elif doc_id in dev_docs:
                w_dev.write(line)
            elif doc_id in test_docs:
                w_test.write(line)

if __name__ == '__main__':
    import sys
    split_data(*sys.argv[1:])
