# Simple Search Engine

## Components

### index

Input: documents

Output: list of (word, doc_id, word_frequency) tuples sorted by (word, doc_id) ASC

### search

Input: query

Output: list of doc_id sorted by tf-idf score DESC

## Dependencies

* nltk
* numpy
