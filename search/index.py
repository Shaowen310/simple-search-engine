import collections
import bisect
import json
import os
import csv
import pickle
from collections.abc import Sequence, MutableSequence, MutableMapping


class DictKeyStr(MutableSequence):
    def __init__(self):
        super().__init__()
        self.keystr = ''
        self.sidx = [0]

    def append(self, item: str):
        """

        Args:
            item (str): string to append
        """
        self.keystr += item
        self.sidx.append(len(self.keystr))

    def __len__(self):
        return len(self.sidx) - 1

    def _check_index(self, key):
        if key < 0 or key >= len(self):
            raise IndexError()

    def __getitem__(self, key):
        self._check_index(key)
        return self.keystr[self.sidx[key]:self.sidx[key + 1]]

    def __setitem__(self, key, item):
        raise AttributeError()

    def __delitem__(self, key):
        raise AttributeError()

    def insert(self, key, item):
        raise AttributeError()


class KeyStrBlock(Sequence):
    def __init__(self, keystr):
        self.keystr = keystr

    def __len__(self):
        c = 0
        itemiter = iter(self)
        try:
            while True:
                next(itemiter)
                c += 1
        except StopIteration:
            return c

    def __iter__(self):
        sid = 0
        cid = 0
        while cid < len(self.keystr):
            ch = self.keystr[cid]
            if ch >= '0' and ch <= '9':
                cid += 1
            else:
                strlen = int(self.keystr[sid:cid])
                yield self.keystr[cid:cid + strlen]
                cid += strlen
                sid = cid

    def __getitem__(self, key):
        itemiter = iter(self)
        item = None
        try:
            for _ in range(key + 1):
                item = next(itemiter)
        except StopIteration:
            raise IndexError()
        return item


class DictKeyStrBlocked(MutableSequence):
    def __init__(self, block_size):
        super().__init__()
        self.keystr = ''
        self.bs = block_size
        self.bsidx = [0]
        self.bkeycnt = 0

    def append(self, item: str):
        self.keystr += str(len(item)) + item
        if self.bkeycnt:
            self.bsidx[-1] = len(self.keystr)
        else:
            self.bsidx.append(len(self.keystr))

        self.bkeycnt = (self.bkeycnt + 1) % self.bs

    def __len__(self):
        return self.bs * (len(self.bsidx) - (2 if self.bkeycnt else 1)) + self.bkeycnt

    def _keystrblk_of(self, bid):
        return KeyStrBlock(self.keystr[self.bsidx[bid]:self.bsidx[bid + 1]])

    def keystriter_of(self, bid):
        return iter(self._keystrblk_of(bid))

    def __iter__(self):
        for bid in range(len(self.bsidx) - 1):
            yield from self.keystriter_of(bid)

    def _check_index(self, key):
        if key < 0 or key >= len(self):
            raise IndexError()

    def __getitem__(self, key):
        self._check_index(key)
        bid = key // self.bs
        sbshift = key % self.bs
        keyblk = self._keystrblk_of(bid)
        return keyblk[sbshift]

    def __setitem__(self, key, item):
        raise AttributeError()

    def __delitem__(self, key):
        raise AttributeError()

    def insert(self, key, item):
        raise AttributeError()


class DictKeyStrBlockedBSView(Sequence):
    def __init__(self, keystrblks):
        self.keystrblks = keystrblks

    def __len__(self):
        return len(self.keystrblks.bsidx) - 1

    def _check_index(self, key):
        if key < 0 or key >= len(self):
            raise IndexError()

    def __getitem__(self, key):
        self._check_index(key)
        return self.keystrblks[key * self.keystrblks.bs]


class DictAsStr(MutableMapping):
    """Requires putting dict keys in sorted order
    """
    def __init__(self):
        super().__init__()
        self.termkeys = DictKeyStr()
        self.postingss = []

    def _find(self, key):
        idx = bisect.bisect_left(self.termkeys, key)
        if idx < len(self.termkeys) and self.termkeys[idx] == key:
            return idx
        return None

    def __contains__(self, key):
        return self._find(key) is not None

    def __getitem__(self, key):
        idx = self._find(key)
        if idx is None:
            raise KeyError()
        return self.postingss[idx]

    def __setitem__(self, key, item):
        idx = self._find(key)
        if idx is None:
            raise KeyError()
        else:
            self.postingss[idx] = item

    def sortedsetdefaultf(self, key, factory):
        keyslen = len(self.termkeys)
        idx = keyslen - 1
        if not keyslen or self.termkeys[idx] != key:
            self.termkeys.append(key)
            self.postingss.append(factory())
            idx += 1
        return self.postingss[idx]

    def __delitem__(self, key):
        raise AttributeError()

    def __iter__(self):
        return iter(self.termkeys)

    def __len__(self):
        return len(self.postingss)

    def keys(self):
        return self.termkeys

    def values(self):
        return self.postingss


class DictAsStrBlocked(DictAsStr):
    def __init__(self, block_size):
        super().__init__()
        self.termkeys = DictKeyStrBlocked(block_size)
        self.termkeysview = DictKeyStrBlockedBSView(self.termkeys)

    def _find(self, key):
        if not len(self.termkeysview):
            return None
        bid = bisect.bisect_right(self.termkeysview, key) - 1
        shift = 0
        termiter = self.termkeys.keystriter_of(bid)
        for term in termiter:
            if term == key:
                return bid * self.termkeys.bs + shift
            shift += 1
        return None


class Index:
    def __init__(self, sortedtdfile, itodfile, rebuild=False, dict_compression='no', block_size=16):
        '''
        Args:
            dict_compression ({'no', 'asstr', 'asstrblk'})
        '''
        self.sortedtdfile = sortedtdfile
        self.itodfile = itodfile

        self.dict_compression = dict_compression
        self.block_size = block_size

        self.dumpfilename = '_'.join(('invidxdump', self.dict_compression)) + '.pkl'
        self.dumpfile = os.path.join('data_', 'invidx', self.dumpfilename)

        if rebuild or not os.path.exists(self.dumpfile):
            self._build()
            self._save(self.dumpfile)
        else:
            self.load(self.dumpfile)

    @staticmethod
    def read_term_doc_freqs(file: str):
        with open(file, 'r') as f:
            csvr = csv.reader(f)
            for row in csvr:
                yield row[0], int(row[1]), int(row[2])

    @staticmethod
    def doctftodf(doctf):
        df = collections.defaultdict(lambda: [])
        for doc, tf in doctf.items():
            for t in tf.keys():
                df[t].append(doc)
        return dict(df)

    def _build(self):
        newpostings = lambda: []
        if self.dict_compression == 'no':
            term_postings = collections.defaultdict(newpostings)
        elif self.dict_compression == 'asstr':
            term_postings = DictAsStr()
        elif self.dict_compression == 'asstrblk':
            term_postings = DictAsStrBlocked(self.block_size)
        else:
            raise NotImplementedError()

        if self.dict_compression == 'no':

            def term_postings_append_tdoc(tdoc):
                term_postings[tdoc[0]].append(tdoc[1])
                return tdoc
        else:

            def term_postings_append_tdoc(tdoc):
                term_postings.sortedsetdefaultf(tdoc[0], newpostings).append(tdoc[1])
                return tdoc

        self.doctf = collections.defaultdict(lambda: {})
        for tdf in Index.read_term_doc_freqs(self.sortedtdfile):
            term_postings_append_tdoc(tdf)
            self.doctf[tdf[1]][tdf[0]] = tdf[2]

        if self.dict_compression == 'no':
            term_postings = dict(term_postings)

        self.invidx = term_postings

        with open(self.itodfile, 'r') as file:
            self.itod = json.load(file)

        self.doctf = dict(self.doctf)
        self.df = __class__.doctftodf(self.doctf)

    def _save(self, fp):
        dumpdir = os.path.dirname(fp)
        if not os.path.exists(dumpdir):
            os.makedirs(dumpdir)
        self.save(fp)

    def save(self, fp):
        with open(fp, 'wb') as file:
            dump = {'invidx': self.invidx, 'itod': self.itod, 'doctf': self.doctf}
            pickle.dump(dump, file)

    def load(self, fp):
        with open(fp, 'rb') as file:
            dump = pickle.load(file)
            self.invidx = dump['invidx']
            self.itod = dump['itod']
            self.doctf = dump['doctf']
        self.df = __class__.doctftodf(self.doctf)
