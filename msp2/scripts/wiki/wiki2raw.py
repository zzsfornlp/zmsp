#

# from wikiextractor's json to zjson

import sys
from msp2.data.inst import Doc
from msp2.data.rw import WriterGetterConf
from msp2.utils import default_json_serializer

def convert(doc):
    ret = Doc.create(text=doc["text"], id=f"wiki{doc['id']}")
    ret.info.update({k:v for k,v in doc.items() if k!="text"})
    return ret

def main(file_in="", file_out=""):
    if file_in in ["", "-"]:
        file_in = sys.stdin
    # --
    with WriterGetterConf().get_writer(output_path=file_out, output_format='zjson_doc') as writer:
        for doc in default_json_serializer.yield_iter(file_in):
            doc2 = convert(doc)
            writer.write_inst(doc2)
    # --

# PYTHONPATH=../?? python3 wiki2raw.py IN OUT
if __name__ == '__main__':
    main(*sys.argv[1:])
