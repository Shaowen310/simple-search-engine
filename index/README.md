# BSBI

Demo for BSBI.

## Dataset

`HillaryEmails`

## Assumptions

Document path as docID

## Parameters

```
--data-dir
--block-size
--outfile
```
path to the directory contining the text files to index
block size parameter for BSBI/SPIMI

Output
Single text file containing a sorted list of term-document pairs
Sort by term

## Modules

### Tokenizer

Split by whitespace and trim, remove empty tokens

### Linguistic

* Remove punctuation
* Remove numbers
* Case folding
* Stemming -> porter stemmer
