import argparse
import os
import json
import time
import datetime
import tracemalloc

from tokenizer import get_docs_in, generate_tokens
from linguistic import generate_normalized_tokens
from loggingutil import get_logger
from bsbi import BSBI

_logger = get_logger('app')


def cli_arg_parser():

    parser = argparse.ArgumentParser(description='Demo for BSBI.')
    parser.add_argument('--data-dir', type=str, default='data_', help='data directory')
    parser.add_argument('--block-size',
                        type=int,
                        default=65536,
                        help='the maximum number of tokens to be processed per block')
    parser.add_argument('--block-dir', type=str, default='block_', help='posting blocks directory')
    return parser


if __name__ == '__main__':
    argparser = cli_arg_parser()
    args = argparser.parse_args()

    tracemalloc.start()

    index_time = time.time()

    id_to_doc = sorted(get_docs_in(args.data_dir))

    tokens = generate_tokens(id_to_doc)
    tokens = generate_normalized_tokens(tokens)

    bsbi = BSBI(args.block_size, args.block_dir)

    with open(os.path.join(args.block_dir, 'id_to_doc.json'), 'w') as file:
        json.dump(id_to_doc, file)

    outfile = bsbi.process(tokens)

    index_time = time.time() - index_time
    _logger.debug('Time to index the dataset: {}'.format(datetime.timedelta(seconds=index_time)))

    _logger.info('Merged index file: {}'.format(outfile))

    current, peak = tracemalloc.get_traced_memory()
    _logger.debug(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    tracemalloc.stop()
