from enum import Enum, unique
from typing import Iterable

import numpy as np

from linguistic import normalize


@unique
class Logical(Enum):
    NOT = 0
    AND = 1
    OR = 2


PRECEDENCE = {Logical.AND: 10, Logical.OR: 11, Logical.NOT: 12}


class QueryToken():
    @unique
    class Type(Enum):
        TERM = 0
        OPERATOR = 1

    def __init__(self, type: Type, content):
        super().__init__()
        self.type = type
        self.content = content

    def __repr__(self):
        return '<{}>'.format(', '.join((str(self.type), str(self.content))))


class QueryParser():
    def __init__(self):
        super().__init__()

    @staticmethod
    def operator_from(text: str):
        if text in Logical.__members__.keys():
            return Logical[text]
        return None

    @staticmethod
    def to_postfix(op, opstack: list, qtokens: list):
        while len(opstack):
            opstacktop = opstack[-1]
            if PRECEDENCE[op] <= PRECEDENCE[opstacktop]:
                qtokens.append(QueryToken(QueryToken.Type.OPERATOR, opstack.pop()))
                continue
            # else PRECEDENCE[op] > PRECEDENCE[opstacktop]
            break
        opstack.append(op)

    @staticmethod
    def parse(query: str):
        opstack = []
        qtokens = []

        tokens = query.strip().split()

        expectop = False

        for node in tokens:
            op = QueryParser.operator_from(node)
            if isinstance(op, Logical) and op is not Logical.NOT:
                if not expectop:
                    raise SyntaxError('Unexpected operator: {}'.format(op.name))
                QueryParser.to_postfix(op, opstack, qtokens)
                expectop = False
            elif op is None or op is Logical.NOT:
                if expectop:
                    # insert implicit AND
                    QueryParser.to_postfix(Logical.AND, opstack, qtokens)
                if op is Logical.NOT:
                    QueryParser.to_postfix(op, opstack, qtokens)
                    expectop = False
                else:  # op is None
                    qtokens.append(QueryToken(QueryToken.Type.TERM, normalize(node)))
                    expectop = True
            else:
                raise NotImplementedError()

        # flush opstack
        while len(opstack):
            qtokens.append(QueryToken(QueryToken.Type.OPERATOR, opstack.pop()))

        return qtokens


class Search():
    def __init__(self, index, skipptr=False):
        super().__init__()
        self.parser = QueryParser()
        self.index = index

    @staticmethod
    def merge_not(pl1: Iterable, pl2: Iterable):
        '''
        Pattern: AND-NOT
        pl1 - pl2
        '''
        result = []
        pl1iter = iter(pl1)
        pl2iter = iter(pl2)
        post1 = next(pl1iter, None)
        post2 = next(pl2iter, None)
        while (post1 is not None) and (post2 is not None):
            if post1 <= post2:
                if post1 == post2:
                    post2 = next(pl2iter, None)
                else:
                    result.append(post1)
                post1 = next(pl1iter, None)
            else:
                # skippable
                post2 = next(pl2iter, None)
        while post1 is not None:
            result.append(post1)
            post1 = next(pl1iter, None)
        return result

    @staticmethod
    def merge_and(pl1: Iterable, pl2: Iterable):
        '''
        Args:
            pl1(Iterable): Posting list 1
            pl2(Iterable): Posting list 2
        '''
        result = []
        pl1iter = iter(pl1)
        pl2iter = iter(pl2)
        post1 = next(pl1iter, None)
        post2 = next(pl2iter, None)
        while (post1 is not None) and (post2 is not None):
            if post1 <= post2:
                if post1 == post2:
                    result.append(post1)
                    post2 = next(pl2iter, None)
                # skippable for post1 < post2 case
                post1 = next(pl1iter, None)
            else:
                # skippable
                post2 = next(pl2iter, None)
        return result

    @staticmethod
    def merge_or(pl1: Iterable, pl2: Iterable):
        result = []
        pl1iter = iter(pl1)
        pl2iter = iter(pl2)
        post1 = next(pl1iter, None)
        post2 = next(pl2iter, None)
        while (post1 is not None) and (post2 is not None):
            if post1 <= post2:
                if post1 == post2:
                    post2 = next(pl2iter, None)
                result.append(post1)
                post1 = next(pl1iter, None)
            else:
                result.append(post2)
                post2 = next(pl2iter, None)
        while post1 is not None:
            result.append(post1)
            post1 = next(pl1iter, None)
        while post2 is not None:
            result.append(post2)
            post2 = next(pl2iter, None)
        return result

    def eval_tf(self, doc, term):
        doctf = self.index.doctf
        if doc not in doctf:
            return 0
        # else
        if term not in doctf[doc]:
            return 0
        # else
        tf = doctf[doc][term]
        return tf

    def eval_rttf(self, result, terms, normalize='max', smoothing=0.4):
        doctf = self.index.doctf
        rttf = np.zeros((len(result), len(terms)), dtype=np.float32)
        for rid, doc in enumerate(result):
            for tid, term in enumerate(terms):
                rttf[rid][tid] = self.eval_tf(doc, term)
            if normalize == 'max':
                if doc in doctf:
                    rttf[rid] /= max(doctf[doc].values())
        if normalize == 'max':
            rttf = smoothing + (1 - smoothing) * rttf
        return rttf

    def eval_tdf(self, terms):
        df = self.index.df
        tdf = np.zeros((len(terms), ), dtype=np.float32)
        for tid, term in enumerate(terms):
            tdf[tid] = len(df[term]) if term in df else 1
        return tdf

    def rank(self, result, qtokens, method='tf-idf'):
        if method is None:
            return result
        # else
        terms = []
        for t in qtokens:
            if t.type is QueryToken.Type.TERM:
                terms.append(t.content)

        rtscores = np.zeros((len(result), len(terms)), dtype=np.float32)  # result term scores
        if method == 'tf':  # term-frequency
            rtscores = self.eval_rttf(result, terms)
        elif method == 'tf-idf':
            tf = self.eval_rttf(result, terms)
            df = self.eval_tdf(terms)
            ndocs = len(self.index.itod)
            idf = np.log(ndocs / df)
            rtscores = tf * idf
        else:
            raise ValueError()

        rscores = np.sum(rtscores, axis=1)

        sorted_idx = np.argsort(-rscores).tolist()

        return [result[i] for i in sorted_idx], [float(rscores[i]) for i in sorted_idx]

    def search(self, query):
        # Parse
        qtokens = self.parser.parse(query)

        # Evaluate
        intermrs = []  # intermrs: intermediate results
        for t in qtokens:
            if t.type is QueryToken.Type.TERM:
                intermrs.append(self.index.invidx.get(t.content, []))
            elif t.type is QueryToken.Type.OPERATOR:
                if isinstance(t.content, Logical):
                    if t.content is Logical.NOT:
                        result = Search.merge_not(list(range(len(self.index.itod))), intermrs.pop())
                        intermrs.append(result)
                    elif t.content is Logical.AND:
                        result = Search.merge_and(intermrs.pop(), intermrs.pop())
                        intermrs.append(result)
                    elif t.content is Logical.OR:
                        result = Search.merge_or(intermrs.pop(), intermrs.pop())
                        intermrs.append(result)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
        result = intermrs.pop()

        # Rank
        result, scores = self.rank(result, qtokens, method='tf-idf')

        return [self.index.itod[docid] for docid in result], scores
