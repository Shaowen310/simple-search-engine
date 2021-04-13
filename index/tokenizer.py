import codecs
import os
import itertools
from typing import Iterable


def list_dir_txt_recur(root: str):
    for path, _, files in os.walk(root):
        for file in files:
            fp = os.path.join(path, file)
            ext = os.path.splitext(fp)[-1].lower()
            if ext != '.txt':
                continue
            yield fp


def get_docs_in(root: str):
    return list(list_dir_txt_recur(root))


def generate_tokens_by_file(fp: str, docid: int):
    with codecs.open(fp, 'r', 'utf-8') as f:
        for line in f:
            tokens = line.split()
            yield from itertools.chain(map(lambda t: (t, docid), tokens))


def generate_tokens(fps: Iterable[str]):
    yield from itertools.chain(*map(lambda idfp: generate_tokens_by_file(idfp[1], idfp[0]), enumerate(fps)))
