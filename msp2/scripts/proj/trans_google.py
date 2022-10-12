#

# using google's api

def translate_text(text: str, src: str, trg: str):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    import six
    from google.cloud import translate_v2 as translate
    # --
    translate_client = translate.Client()
    # --
    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")
    # --
    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, source_language=src, target_language=trg, format_='text', model='nmt')
    # --
    # print(u"Text: {}".format(result["input"]))
    # print(u"Translation: {}".format(result["translatedText"]))
    # print(u"Detected source language: {}".format(result["detectedSourceLanguage"]))
    return result

# --
def main(src: str, trg: str, f_input: str, f_output: str):
    import json, os, traceback
    assert not os.path.exists(f_output)
    with open(f_input) as fd, open(f_output, 'w') as fd2:
        for line in fd:
            line = line.strip()
            if len(line) > 0:
                try:
                    res = translate_text(line, src, trg)
                except:
                    err = traceback.format_exc()
                    res = {"input": line, "translatedText": None, "err": err}
            fd2.write(json.dumps(res, ensure_ascii=False)+"\n")
            fd2.flush()
    # --

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

# --
# GOOGLE_APPLICATION_CREDENTIALS=_key/_key.json python3 trans_google.py en ?? {IN} {OUT}
# for cl in de; do GOOGLE_APPLICATION_CREDENTIALS=_key/_key.json python3 trans_google.py en ${cl} en.ewt.txt _trans.en-${cl}.raw.json; done
# --
# note: before adding "format=text", need to unescape
# python3 -c "import json,sys,html; print(''.join([html.unescape(json.loads(line)['translatedText'])+'\n' for line in sys.stdin]),end='')"
# --
# for cl in de fr it es pt 'fi'; do python3 -c "import json, sys; print(''.join([json.loads(line)['translatedText']+'\n' for line in sys.stdin]),end='')" <_trans.en-${cl}.raw.json >_trans.en-${cl}.raw.txt; done
