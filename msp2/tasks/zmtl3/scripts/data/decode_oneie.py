import os
import json
import glob
import tqdm
import traceback
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig

from model import OneIE
from config import Config
from util import save_result
from scorer import score_graphs, convert_arguments
from data import IEDatasetEval, IEDataset
from convert import json_to_cs
from graph import Graph

cur_dir = os.path.dirname(os.path.realpath(__file__))
format_ext_mapping = {'txt': 'txt', 'ltf': 'ltf.xml', 'json': 'json',
                      'json_single': 'json'}

def load_model(model_path, device=0, gpu=False, beam_size=5):
    print('Loading the model from {}'.format(model_path))
    map_location = 'cuda:{}'.format(device) if gpu else 'cpu'
    state = torch.load(model_path, map_location=map_location)

    config = state['config']
    if type(config) is dict:
        config = Config.from_dict(config)
    config.bert_cache_dir = os.path.join(cur_dir, 'bert')
    vocabs = state['vocabs']
    valid_patterns = state['valid']

    # recover the model
    model = OneIE(config, vocabs, valid_patterns)
    model.load_state_dict(state['model'])
    model.beam_size = beam_size
    if gpu:
        model.cuda(device)

    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name,
                                              cache_dir=config.bert_cache_dir,
                                              do_lower_case=False)

    return model, tokenizer, config

def load_pred_graph(d: dict, vocabs):
    voc_ef, voc_evt, voc_rel, voc_arg = [vocabs[z] for z in ['entity_type', 'event_type', 'relation_type', 'role_type']]
    entities = [[i, j, voc_ef[k]] for i,j,k,*z in d['pred']['entities']]
    triggers = [[i, j, voc_evt[k]] for i,j,k,*z in d['pred']['triggers']]
    relations = [[i, j, voc_rel[k]] for i,j,k,*z in d['pred']['relations']]
    roles = [[i, j, voc_arg[k]] for i,j,k,*z in d['pred']['roles']]
    g = Graph(entities, triggers, relations, roles, vocabs)
    return g

def predict(model_path, input_path, output_path, eval_path, log_path=None, cs_path=None,
         batch_size=50, max_length=128, device=0, gpu=False,
         file_extension='txt', beam_size=5, input_format='txt',
         language='english'):
    """Perform information extraction.
    :param model_path (str): Path to the pre-trained model file.
    :param input_path (str): Path to the input directory.
    :param output_path (str): Path to the output directory.
    :param log_path (str): Path to the log file.
    :param cs_path (str): (optional) Path to the cold-start format output directory.
    :param batch_size (int): Batch size (default=50).
    :param max_length (int): Max word piece number for each sentence (default=128).
    :param device (int): GPU device index (default=0).
    :param gpu (bool): Use GPU (default=False).
    :param file_extension (str): Input file extension. Only files ending with the
    given extension will be processed (default='txt').
    :param beam_size (int): Beam size of the decoder (default=5).
    :param input_format (str): Input file format (txt or ltf, default='txt').
    :param language (str): Document language (default='english').
    """
    # set gpu device
    if gpu:
        torch.cuda.set_device(device)
    # load the model from file
    model, tokenizer, config = load_model(model_path, device=device, gpu=gpu,
                                          beam_size=beam_size)
    # --
    test_set = IEDataset(input_path, gpu=gpu,
                         relation_mask_self=config.relation_mask_self,
                         relation_directional=config.relation_directional,
                         symmetric_relations=config.symmetric_relations)
    test_set.numberize(tokenizer, model.vocabs)
    # --
    # simply do eval!
    if eval_path:
        with open(eval_path) as fd:
            eval_graphs = []
            for line in fd:
                d = json.loads(line)
                g = load_pred_graph(d, model.vocabs)
                eval_graphs.append(g)
        gold_graphs = []
        for batch in DataLoader(test_set, batch_size=config.eval_batch_size,
                                shuffle=False, collate_fn=test_set.collate_fn):
            gold_graphs.extend(batch.graphs)
        # --
        # # debug
        # debug_cc = [len(convert_arguments(g.triggers, g.entities, g.roles)) for g in gold_graphs]
        # print(debug_cc)
        # --
        score_graphs(gold_graphs, eval_graphs, relation_directional=config.relation_directional)
        return
    # --
    test_batch_num = len(test_set) // config.eval_batch_size + \
                     (len(test_set) % config.eval_batch_size != 0)
    progress = tqdm.tqdm(total=test_batch_num, ncols=75, desc='Test')
    test_gold_graphs, test_pred_graphs, test_sent_ids, test_tokens = [], [], [], []
    for batch in DataLoader(test_set, batch_size=config.eval_batch_size,
                            shuffle=False, collate_fn=test_set.collate_fn):
        progress.update(1)
        graphs = model.predict(batch)
        for graph in graphs:
            graph.clean(relation_directional=config.relation_directional,
                        symmetric_relations=config.symmetric_relations)
        test_gold_graphs.extend(batch.graphs)
        test_pred_graphs.extend(graphs)
        test_sent_ids.extend(batch.sent_ids)
        test_tokens.extend(batch.tokens)
    progress.close()
    test_scores = score_graphs(test_gold_graphs, test_pred_graphs,
                               relation_directional=config.relation_directional)
    save_result(output_path, test_gold_graphs, test_pred_graphs,
                test_sent_ids, test_tokens)
    # --

parser = ArgumentParser()
parser.add_argument('-m', '--model_path', help='path to the trained model')
parser.add_argument('-i', '--input_dir', help='path to the input folder (ltf files)')
parser.add_argument('-o', '--output_dir', help='path to the output folder (json files)')
parser.add_argument('-e', '--eval_path', help='path for the eval file')
parser.add_argument('-l', '--log_path', default=None, help='path to the log file')
parser.add_argument('-c', '--cs_dir', default=None, help='path to the output folder (cs files)')
parser.add_argument('--gpu', action='store_true', help='use gpu')
parser.add_argument('-d', '--device', default=0, type=int, help='gpu device index')
parser.add_argument('-b', '--batch_size', default=10, type=int, help='batch size')
parser.add_argument('--max_len', default=128, type=int, help='max sentence length')
parser.add_argument('--beam_size', default=5, type=int, help='beam set size')
parser.add_argument('--lang', default='english', help='Model language')
parser.add_argument('--format', default='txt', help='Input format (txt, ltf, json)')

args = parser.parse_args()
extension = format_ext_mapping.get(args.format, 'ltf.xml')

predict(
    model_path=args.model_path,
    input_path=args.input_dir,
    output_path=args.output_dir,
    eval_path=args.eval_path,
    cs_path=args.cs_dir,
    log_path=args.log_path,
    batch_size=args.batch_size,
    max_length=args.max_len,
    device=args.device,
    gpu=args.gpu,
    beam_size=args.beam_size,
    file_extension=extension,
    input_format=args.format,
    language=args.lang,
)

# python3 decode_oneie.py ...
