#

# use MediaWiki api
# note: from https://stackoverflow.com/questions/4452102/how-to-get-plain-text-out-of-wikipedia
# note: or simply use Wikipedia? https://wikipedia.readthedocs.io/en/latest/ https://github.com/goldsmith/Wikipedia

import sys
from msp2.data.inst import Doc, Sent
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.utils import default_json_serializer, Conf

class MainConf(Conf):
    def __init__(self):
        self.W = WriterGetterConf()
        self.lang = "en"
        self.method = "html"  # html or api
        # --
        self.titles = []
        self.pageids = []
        # use html-page
        # use wiki-api
        self.exintro = False
        self.explaintext = True

# api
def get_by_api(conf: MainConf):
    import requests
    for title, pageid in [(t, None) for t in conf.titles] + [(None, p) for p in conf.pageids]:
        params = {'action': 'query', 'format': 'json', 'prop': 'extracts'}
        if title:
            params["titles"] = title
        if pageid:
            params["pageids"] = pageid
        if conf.exintro:
            params['exintro'] = True
        if conf.explaintext:
            params['explaintext'] = True
        response = requests.get(f'https://{conf.lang}.wikipedia.org/w/api.php', params=params).json()
        for val in response['query']['pages'].values():
            _id, title, text = f"wiki{val['pageid']}", val['title'], val['extract']
            doc = Doc.create(text=text, id=_id)
            doc.info['title'] = title
            yield doc
    # --

# html
def get_by_html(conf: MainConf):
    import requests
    import re
    from bs4 import BeautifulSoup
    for title, pageid in [(t, None) for t in conf.titles] + [(None, p) for p in conf.pageids]:
        if title:
            url = f"https://{conf.lang}.wikipedia.org/wiki/{title}"
        elif pageid:
            url = f"https://{conf.lang}.wikipedia.org/?curid={pageid}"
        else:
            raise RuntimeError()
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        # --
        actual_title = soup.select("#firstHeading")[0].text
        texts = []
        for p in soup.find_all('p'):
            one_text = re.sub(r'\[.*?\]', '', p.text)
            texts.append(one_text)
        doc = Doc.create(text="\n".join(texts))
        doc.info.update({'title': actual_title, 'pageid': pageid})
        yield doc

# --
def main(*args):
    conf = MainConf()
    conf.update_from_args(list(args))
    # --
    ff = {'html': get_by_html, 'api': get_by_api}[conf.method]
    with conf.W.get_writer() as writer:
        for doc in ff(conf):
            writer.write_inst(doc)
            # breakpoint()
    # --

# PYTHONPATH=../?? python3 wiki_request.py titles:??
if __name__ == '__main__':
    main(*sys.argv[1:])
