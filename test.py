#!/usr/bin/env python

from bpemb import BPEmb
embedding = BPEmb(model_file='en-50k-100d.model', emb_file='en-50k-100d-vectors.txt')

import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file='en-50k-100d.model')


vocab = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]
for v, k in zip(vocab, embedding.emb.index_to_key):
    if v != k:
        print('error:', v, '!=', k)


"""
vocab = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]
for v in vocab:
    try:
        emb = embedding.embed(v)
        if v == '💘':
            print('💘', emb)
    except:
        print('error:', v)
"""

"""
for i in range(sp.get_piece_size()):
    try:
        emb = embedding.emb.vectors[i]
    except:
        print('error:', sp.id_to_piece(i))
"""

"""
test_string = 'you are amazing and talented and i hope you have an amazing night 💘💕💖💘'
print(test_string)
print(embedding.encode_ids(test_string))
embedding.embed(test_string)
"""
