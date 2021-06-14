#!/usr/bin/env python

import re
import os
import sys
import html
import unicodedata
import sentencepiece as spm
from glob import glob
from dataload.loader import *
from dataload.preproc import *

"""
train custom bpe embeddings

usage
- set parameter: language, model name, vocabulary size
    - optional: glove parameter see glove/demo.sh
- load resulting files in bpemb
    - BPEmb(model_file='en-custom-10k.model', emb_file='en-custom-10k-vectors.txt', dim=300)

"""

LANGUAGE = 'de'
MODEL_NAME = 'de-50k-300d'
VOCAB_SIZE = 50000
EMBEDDING_DIM = 300  # must also be set in glove/demo.sh
LOWERCASE = True


def load_data_en():
    DATA_1 = 'data/aggression/hate-speech-and-offensive-language/data/labeled_data.csv'  # 24783 items
    DATA_2_TXT = 'data/aggression/AutomaticDetectionofHateSpeech/hatespeech.csv'  # 16035 items
    DATA_2_ANN = 'data/aggression/hatespeech/NAACL_SRW_2016.csv'
    DATA_3_ANN = 'data/aggression/hate-speech-dataset/annotations_metadata.csv'
    DATA_3_TXT = 'data/aggression/hate-speech-dataset/all_files'  # 10944 items
    DATA_4_TXT = 'data/aggression/OffensEval-2019/offenseval-training-v1.tsv'  # 13240 items
    DATA_4_CORR = 'data/aggression/re-labelings/OffensEval-2019-profanity-predicted-edit.csv'
    DATA_5_TXT = 'data/aggression/founta-corpus/hatespeechtwitter-tweets.csv'  # 80.000 items
    DATA_5_ANN = 'data/aggression/founta-corpus/hatespeechtwitter.tsv'
    DATA_5_CORR = 'data/aggression/re-labelings/founta-corpus-profanity-predicted-edit.csv'
    DATA_7 = 'data/aggression/HateXplain/HateXplain/Data/dataset.json'
    DATA_8_1_TXT = 'data/aggression/OffensEval-2020/SOLID/semeval_test/test_a_tweets.tsv'
    DATA_8_1_ANN = 'data/aggression/OffensEval-2020/SOLID/semeval_test/test_a_labels.csv'
    DATA_8_2_TXT = 'data/aggression/OffensEval-2020/SOLID/extended_test/test_a_tweets_all.tsv'
    DATA_8_2_ANN = 'data/aggression/OffensEval-2020/SOLID/extended_test/test_a_labels_all.csv'
    DATA_8_CORR = 'data/aggression/re-labelings/SOLID-testdata-profanity-predicted-edit.csv'
    DATA_9_TXT = 'data/aggression/OLIDv1.0/testset-levela.tsv'
    DATA_9_ANN = 'data/aggression/OLIDv1.0/labels-levela.csv'
    DATA_THREAT = 'data/aggression/hexis-english-threats/hexis-english-threats.txt'
    SPAM_1 = 'data/spam/smsspamcollection/SMSSpamCollection.txt'
    DEBIAS_MADLIBS = 'data/social-bias/selection/bias_madlibs_89k.csv'
    TWITTER_1 = 'examples/aggression_ny/data_171219/twitter_aggression_data.csv'
    TWITTER_2 = 'examples/aggression_ny/data_291219/twitter_aggression_data.csv'
    TWITTER_3 = 'scraping/twitter/twitter_scrape_ny_state/data_130421/merged.csv'
    CORRECTIONS = [DATA_4_CORR, DATA_5_CORR, DATA_8_CORR]

    data = []
    data += hate1(files=[DATA_1], local=True)
    data += hate2(files=[DATA_2_TXT, DATA_2_ANN], local=True)
    data += hate3(files=[DATA_3_ANN, DATA_3_TXT], local=True)
    data += hate4(files=[DATA_5_TXT, DATA_5_ANN], local=True)
    data += offense19(files=[DATA_4_TXT], local=True)
    data += offense20(files=[DATA_8_1_TXT, DATA_8_1_ANN, DATA_8_2_TXT, DATA_8_2_ANN], local=True)
    data += olid(files=[DATA_9_TXT, DATA_9_ANN], local=True)
    data += hatexplain(files=[DATA_7], local=True)
    data += germeval(files=[DATA_THREAT], local=True)
    data += spam1(files=[SPAM_1], local=True)
    data += debias_madlibs(files=[DEBIAS_MADLIBS], local=True)
    data += twint(files=[TWITTER_1], local=True)
    data += twint(files=[TWITTER_2], local=True)
    data += relabel(files=CORRECTIONS, local=True)

    return data


def load_data_de():
    DATA2018_TRAIN = 'data/aggression/GermEval-2018-Data/germeval2018.training.txt'
    DATA2018_TEST = 'data/aggression/GermEval-2018-Data/germeval2018.test.txt'  # 3532 items
    DATA2018_EXTRA = 'data/aggression/germeval2018-upinf/labeled_tweets-hexis.tsv'  # 100 items
    DATA2019_TRAIN_1 = 'data/aggression/GermEval-2019-Data/germeval2019_training_subtask12.txt'
    DATA2019_TRAIN_2 = 'data/aggression/GermEval-2019-Data/germeval2019.training_subtask3.txt'
    DATA2019_TEST_1 = 'data/aggression/GermEval-2019-Data/germeval2019GoldLabelsSubtask12.txt'  # 3031 items
    DATA2019_TEST_2 = 'data/aggression/GermEval-2019-Data/germeval2019GoldLabelsSubtask3.txt'
    DATA_AFDFPOE = 'data/aggression/afd-fpoe-corpus/afd-fpoe-corpus.txt'
    GERMAN_SPAM = 'data/spam/german-spam/german-spam.txt'
    TWITTER = 'examples/aggression_dach/deduped/twitter_aggression_data.csv'
    TWITTER_2 = 'data/aggression/dach-twitter-corpus/src/twitter_scrape_deduplicated.csv'
    DEBIAS_MADLIBS = 'data/social-bias/selection/bias_madlibs_89k-German.csv'
    ALL_GERMEVAL = [DATA2018_TRAIN, DATA2018_TEST, DATA2018_EXTRA, DATA2019_TRAIN_1, DATA2019_TEST_1, DATA2019_TRAIN_2, DATA2019_TEST_2, DATA_AFDFPOE, GERMAN_SPAM]

    data = []
    data += germeval(files=ALL_GERMEVAL, local=True)
    data += twint([TWITTER], local=True)
    data += twint([TWITTER_2], local=True)
    data += debias_madlibs(files=[DEBIAS_MADLIBS], local=True)

    return data


class SentenceIterator():
    def __init__(self, data):
        self.data = data
        self.index = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            entry = self.data[self.index]
        except IndexError:
            raise StopIteration
        self.index += 1
        
        if isinstance(entry['text'], str):
            text = clean_str(entry['text'])
            #text = clean_str_mini(entry['text'])  # alternative minimal version
            if text != '':
                if LOWERCASE == True:
                    return text.lower()
                else:
                    return text
            else:
                return ''
        else:
            return ''


if __name__ == '__main__':
    
    print('* loading data')
    if LANGUAGE == 'de':
        data = load_data_de()
    elif LANGUAGE == 'en':
        data = load_data_en()
    else:
        print('language not found')

    
    print('* training bpe model')
    # test: hard_vocab_limit=False, input_sentence_size=500000
    spm.SentencePieceTrainer.train(sentence_iterator=SentenceIterator(data), model_prefix=MODEL_NAME, model_type='bpe', vocab_size=VOCAB_SIZE)
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
