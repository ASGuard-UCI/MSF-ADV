'''
Evaluate a model on a dataset using a evaluator.
'''

import logging
import os
import argparse
import torch
import yaml
import torch
from torch.utils import data

import config
from global_var import param
import global_var
from utils import StoreDictKeyPair, registry

import models as _models
import dataset as _dataset
import evaluator as _evaluator


def evaluate(debug):
    # Load the network, dataset according to the parameter file specified using command line
    model = registry.create('Network', param.network.name)(**param.network.kwargs)
    model.eval()

    dataset = registry.create('Dataset', param.dataset.name)(**param.dataset.kwargs)
    evaluator = registry.create('Evaluator', param.evaluator.name)(**param.evaluator.kwargs)
    summaries = evaluator(model, dataset)

    print(summaries['scalar'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_params', '-lp', default='eval_render4cnn_pascal3dplus', help='The global parameters for the program to load, from param/ folder')
    parser.add_argument('--overwrite_param', '-op', action=StoreDictKeyPair, default={}, nargs='+', help='The parameters to override')
    parser.add_argument('--debug', action='store_true', default=False, help='The debugging flag')

    args = parser.parse_args()

    # In debugging mode, we always use `debug` as the identification string
    if args.debug:
        config.UNIQUE_ID = 'debug'

    # Setup logging
    log_format = '%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'

    if args.debug: # Show more verbose info if debugging
        logging.basicConfig(level=logging.DEBUG, format=log_format)
    else:          # Show less info
        logging.basicConfig(level=logging.INFO, format=log_format)

    # Load the global parameters
    global_var.param_load_yaml(os.path.join('param', '{}.yml'.format(args.load_params)))
    # logging.info(args_str)

    # And update override the specified parameters
    param.update(args.overwrite_param)

    # Also dump the parameters for possible future reproduction
    print('================param================')
    to_print = yaml.dump(param.to_dict(), default_flow_style=False)
    print(to_print)
    print('-------------------------------------')

    evaluate(args.debug)
