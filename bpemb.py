#!/usr/bin/env python

import re
import os
import sys
import html
import unicodedata
import sentencepiece as spm
from glob import glob

"""
train custom bpe embeddings

usage
- set parameter: model name, vocabulary size
    - optional: glove parameter see glove/demo.sh
- load resulting files in bpemb
    - BPEmb(model_file='en-custom-10k.model', emb_file='en-custom-10k-vectors.txt', dim=300)

"""

MODEL_NAME = 'de-50k-300d'
INPUT_PATH = 'corpus.txt'
VOCAB_SIZE = 50000
EMBEDDING_DIM = 300  # must also be set in glove/demo.sh
LOWERCASE = True


class SentenceIterator:
    def __init__(self, file_path, encoding="utf-8"):
        self.file = open(file_path, "r", encoding=encoding)

    def __iter__(self):
        return self

    def __next__(self):
        line = self.file.readline()
        if not line:  # eof
            self.file.close()
            raise StopIteration
        return line.strip()


if __name__ == '__main__':
    
    print('* training bpe model')
    # test: hard_vocab_limit=False, input_sentence_size=500000
    spm.SentencePieceTrainer.train(sentence_iterator=SentenceIterator(INPUT_PATH), model_prefix=MODEL_NAME, model_type='bpe', vocab_size=VOCAB_SIZE)
    print('** model saved to '+MODEL_NAME+'.model')
    

    print('* encoding corpus using bpe model')
    sp = spm.SentencePieceProcessor(model_file=MODEL_NAME+'.model')
    #data_encoded = [sp.encode(entry, out_type=str) for entry in SentenceIterator(data)]  # string encoding
    #data_encoded = [sp.encode(entry, out_type=int) for entry in SentenceIterator(data)]  # integer encoding

    data_encoded = []
    for entry in SentenceIterator(data):        
        for n in range(10):  # sampling to elicit all pieces
            entry_encoded = sp.encode(entry, out_type=str, enable_sampling=True, alpha=0.1)
            data_encoded.append(entry_encoded)

    with open(MODEL_NAME+'.txt', 'w') as f:
        for entry in data_encoded:
            f.write(' '.join(entry) + '\n')
            #f.write(' '.join([str(e) for e in entry]) + '\n')  # integer encoding
    print('** corpus saved to '+MODEL_NAME+'.txt')


    print('* training embedding')
    os.system('cp '+MODEL_NAME+'.txt glove/corpus.txt') 
    #os.system('cd glove && build/vocab_count -min-count 0 -verbose 2 < corpus.txt > vocab.txt')  # check vocab count only
    os.system('cd glove && ./demo.sh')
    os.system('cp glove/vectors.txt '+MODEL_NAME+'-vectors.txt')
    os.system('rm glove/vectors.txt glove/corpus.txt glove/vocab.txt glove/cooccurrence.bin glove/cooccurrence.shuf.bin')
    print('** embedding saved to '+MODEL_NAME+'-vectors.txt')
    

    print('* aligning embedding')
    sp = spm.SentencePieceProcessor(model_file=MODEL_NAME+'.model')
    vocab = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]
    with open(MODEL_NAME+'-vectors.txt', 'r') as f:
        embedding = f.readlines()

    for line in embedding:
        if line.startswith('<unk>'):
            unk = line
    
    with open(MODEL_NAME+'-vectors-tmp.txt', 'w') as f:
        f.write(str(len(vocab))+' '+str(EMBEDDING_DIM)+'\n')  # header
        f.write(unk)
        f.write(unk.replace('<unk>', '<s>'))  # reinstate sentencepiece special chars
        f.write(unk.replace('<unk>', '</s>'))

        for piece in vocab:
            if piece in ['<unk>', '<s>', '</s>']:
                continue
            found = False
            for line in embedding:
                if line.startswith(piece+' '):  # exact match
                    f.write(line)
                    found = True
            if not found:
                for line in embedding:
                    if line.startswith(piece):  # partial match
                        piece_orig = line.split(' ')[0]
                        line = line.replace(piece_orig, piece)
                        f.write(line)
                        found = True
                if not found:
                    line = unk.replace('<unk>', piece)  # unknown token
                    f.write(line)
    
    print('checking for empty lines ..')
    os.system('grep -E --line-number --with-filename \'^$\' '+MODEL_NAME+'-vectors-tmp.txt')

    os.system('mv '+MODEL_NAME+'-vectors-tmp.txt '+MODEL_NAME+'-vectors.txt')
    print('** embedding saved to '+MODEL_NAME+'-vectors.txt')
