import argparse
import time
import tracemalloc

from index import Index
from search import Search
from loggingutil import get_logger

_logger = get_logger('app')


def cli_arg_parser():

    parser = argparse.ArgumentParser(description='Demo for BSBI.')
    parser.add_argument('--index-file',
                        type=str,
                        default='data_/87.csv',
                        help='data file containing sorted term-document pairs')
    parser.add_argument('--id-to-doc-file',
                        type=str,
                        default='data_/id_to_doc.json',
                        help='document id to document path map')
    return parser


if __name__ == '__main__':
    argparser = cli_arg_parser()
    args = argparser.parse_args()

    index = Index(args.index_file, args.id_to_doc_file, dict_compression='no')

    search = Search(index)

    result, scores = search.search('clinton bush obama biden china OR russia NOT japan')

    rs = [(result[i], scores[i]) for i in range(len(result))]

    _logger.debug(str(rs))
