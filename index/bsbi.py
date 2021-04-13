import os
import shutil
import collections
import itertools
import csv
import queue
import time
import datetime
import sys

from loggingutil import get_logger


class IntIDGenerator:
    def __init__(self, start=0):
        self.start = start

    def __iter__(self):
        self.id = self.start - 1
        return self

    def __next__(self):
        self.id += 1
        return self.id


class BSBI:
    '''
    Blocked sort-based indexing
    
    Reference: 
        https://nlp.stanford.edu/IR-book/html/htmledition/blocked-sort-based-indexing-1.html
    '''
    id_generator = iter(IntIDGenerator())
    _logger = get_logger('bsbi')

    class BlockFileWriter:
        def __init__(self, fostream):
            self.csvw = csv.writer(fostream)

        def writerow(self, row):
            self.csvw.writerow(row)

    def __init__(self, block_size, block_dir='block_'):
        if not (sys.version_info[0] >= 3 and sys.version_info[1] >= 7):
            # dict preserves insertion order
            # SimpleQueue
            raise RuntimeError('Python version >= 3.7 required.')

        self.block_size = block_size
        self.block_dir = block_dir

        # Clean old block files
        if os.path.exists(block_dir) and os.path.isdir(block_dir):
            shutil.rmtree(block_dir)

        os.makedirs(block_dir)

    def get_block(self, token_stream):
        return list(itertools.islice(token_stream, self.block_size))

    @staticmethod
    def bsbi_invert(term_docs):
        sorted_pairs = sorted(term_docs, key=lambda x: (x[0], x[1]))

        # Assume python version >= 3.7, dict preserves insertion order
        term_postings = collections.defaultdict(lambda: [])
        for term_doc in sorted_pairs:
            postings = term_postings[term_doc[0]]
            if not len(postings) or postings[-1][0] != term_doc[1]:
                postings.append([term_doc[1], 1])
            else:
                postings[-1][1] += 1

        return dict(term_postings)

    @staticmethod
    def generate_term_doc_fs(postinglists):
        for (term, docs) in postinglists.items():
            for docfreq in docs:
                yield (term, docfreq[0], docfreq[1])

    def get_block_fp(self, block_id):
        return os.path.join(self.block_dir, str(block_id)) + '.csv'

    def write_block(self, block_id, postinglists):
        term_doc_fs = __class__.generate_term_doc_fs(postinglists)

        file = self.get_block_fp(block_id)
        with open(file, 'w') as f:
            w = self.BlockFileWriter(f)
            for row in term_doc_fs:
                w.writerow(row)

    def read_block(self, block_id):
        file = self.get_block_fp(block_id)
        with open(file, 'r') as f:
            csvr = csv.reader(f)
            for row in csvr:
                yield row[0], int(row[1]), int(row[2])

    def merge_blocks(self, block_queue):
        while block_queue.qsize() > 1:
            block_id0 = block_queue.get()
            block_id1 = block_queue.get()
            block_id2 = next(self.id_generator)
            self.merge_two_blocks(block_id0, block_id1, block_id2)
            block_queue.put(block_id2)

        return block_queue.get()

    def merge_two_blocks(self, block_id0, block_id1, block_id2):
        self._logger.debug('Merging block {} and {} to {}'.format(block_id0, block_id1, block_id2))

        tdf_stream0 = self.read_block(block_id0)
        tdf_stream1 = self.read_block(block_id1)

        outfile = self.get_block_fp(block_id2)
        with open(outfile, 'w') as fout:
            writer = self.BlockFileWriter(fout)

            # merge sort
            tdf0 = next(tdf_stream0, None)
            tdf1 = next(tdf_stream1, None)
            while (tdf0 is not None) and (tdf1 is not None):
                term_doc0 = (tdf0[0], tdf0[1])
                term_doc1 = (tdf1[0], tdf1[1])
                if term_doc0 <= term_doc1:
                    if term_doc0 == term_doc1:
                        tdf0 = (tdf0[0], tdf0[1], tdf0[2] + tdf1[2])
                        tdf1 = next(tdf_stream1, None)
                    writer.writerow(tdf0)
                    tdf0 = next(tdf_stream0, None)
                else:
                    writer.writerow(tdf1)
                    tdf1 = next(tdf_stream1, None)
            while tdf0 is not None:
                writer.writerow(tdf0)
                tdf0 = next(tdf_stream0, None)
            while tdf1 is not None:
                writer.writerow(tdf1)
                tdf1 = next(tdf_stream1, None)

    def process(self, token_stream):
        block_queue = queue.SimpleQueue()
        # write blocks
        while True:
            block_id = next(self.id_generator)
            self._logger.debug('Processing block {}'.format(block_id))

            tokens = self.get_block(token_stream)
            if not len(tokens):
                break

            sort_time = time.time()
            postinglists = __class__.bsbi_invert(tokens)
            sort_time = time.time() - sort_time
            self._logger.debug('Time to sort a block time: {}'.format(
                datetime.timedelta(seconds=sort_time)))
            self.write_block(block_id, postinglists)
            block_queue.put(block_id)

        # merge blocks
        merge_time = time.time()
        merged_block_id = self.merge_blocks(block_queue)
        merge_time = time.time() - merge_time
        self._logger.debug('Time to merge all blocks: {}'.format(
            datetime.timedelta(seconds=merge_time)))

        merged_file = self.get_block_fp(merged_block_id)

        return merged_file
