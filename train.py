import argparse
import collections
import json
import os
import pickle

import pandas as pd
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

data_folder = '../data'
def get_file(path):
    return os.path.join(data_folder, path)

def token_path(path):
    competition, problem, submissions, submission_id = path.split('/')
    return f'tokens/{competition}/{competition}.{problem}.{submission_id}&tokens.csv'

def path_to_problem(path):
    c, p, _, _ = path.split('/')
    return 'codeforces-distinct/' + '/'.join((c, p))

def main(config):
    logger = config.get_logger('train')

    index = pd.read_csv(get_file('index_clean.csv'))
    index = index.sort_values(by=['contest_id'])
    train_indices = index.contest_id <= 800
    test_indices = np.logical_and(800 < index.contest_id, index.contest_id <= 1000)
    validation_indices = 1000 < index.contest_id
    train_paths = index.path[train_indices]
    test_paths = index.path[test_indices]
    validation_paths = index.path[validation_indices]
    train_token_paths = index.token_path[train_indices].map(get_file)
    test_token_paths = index.token_path[test_indices].map(get_file)
    validation_token_paths = index.token_path[validation_indices].map(get_file)

    token_counts = pickle.load(open('data/token_counts.pkl', 'rb'))
    token_tag_counts = pickle.load(open('data/token_tag_counts.pkl', 'rb'))

    problems = set()
    for path in index.path:
        problems.add(path_to_problem(path))

    tags = {}
    all_tags = set()
    for problem in problems:
        meta = json.load(open(get_file(os.path.join(problem, 'meta.json')), 'r'))
        tags[problem] = meta['tags']
        for tag in tags[problem]:
            all_tags.add(tag)

    # token_counts = collections.Counter()
    # for filename in token_paths:
    #     if os.stat(filename).st_size == 0:
    #         continue
    #     token_count = pd.read_csv(filename)
    #     for token, count in zip(token_count['token'], token_count['count']):
    #         token_counts[token] += count

    token2id = {}
    for i, (token, _) in enumerate(token_counts.most_common()):
        token2id[token] = i

    tag2index = {}
    for i, tag in enumerate(all_tags):
        tag2index[tag] = i

    # setup data_loader instances
    data_loader: module_data.TokenDataProducer = config.init_obj('data_loader', module_data)
    data_loader.set_initial_information(train_token_paths, token2id, token_counts)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    trainer = Trainer(model, criterion=None, metric_ftns=[], optimizer=optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=None,
                      lr_scheduler=None, len_epoch=8000)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
