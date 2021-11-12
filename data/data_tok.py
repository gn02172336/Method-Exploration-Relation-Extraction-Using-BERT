import os
import tokenization
import numpy as np
import json
from tqdm import tqdm
from transformers import *


kVocabFile = "./dataset/cased_L-12_H-768_A-12/vocab.txt"
kBookFlag = True

# Modify below three to your corresponding TACRED data location
train = './dataset/tacred/train.json'
dev = './dataset/tacred/dev.json'
test = './dataset/tacred/test.json'


def transform( data, vocabfile ):
    #tokenizer = tokenization.FullTokenizer( vocab_file=vocabfile, do_lower_case=False )
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    mapping = { '-LRB-': '(',
                '-RRB-': ')',
                '-LSB-': '[',
                '-RSB-': ']',
                '-LCB-': '{',
                '-RCB-': '}',
                    0: '[UNK]',
                    1: '[PAD]' }

    for d in tqdm(data):
        tokens = d['token']
        s1 = d['subj_start']
        e1 = d['subj_end']
        s2 = d['obj_start']
        e2 = d['obj_end']

        bert_tokens = []
        orig_to_tok_map = []
        tok_len_map = []
        bert_tokens.append( "[CLS]" )
        for token in tokens:
            if token in mapping:
                token = mapping[token]

            orig_to_tok_map.append( len(bert_tokens) )
            bert_tokens.extend( tokenizer.tokenize(token) )
            tok_len_map.append( len(bert_tokens)-orig_to_tok_map[-1] )

        orig_to_tok_map.append( len(bert_tokens) )


        bs1 = orig_to_tok_map[s1]
        be1 = orig_to_tok_map[e1+1]
        bs2 = orig_to_tok_map[s2]
        be2 = orig_to_tok_map[e2+1]

        d['subj_start'] = bs1
        d['subj_end'] = be1
        d['obj_start'] = bs2
        d['obj_end'] = be2

        bert_tokens.append( "[SEP]" )
        bert_tokens += bert_tokens[bs1:be1]
        bert_tokens.append( "[SEP]" )
        bert_tokens += bert_tokens[bs2:be2]
        bert_tokens.append( "[SEP]" )

        d['token'] = bert_tokens
        
        if not kBookFlag:

            spos = d['stanford_pos']
            spos2 = ['[CLS]']
            sner = d['stanford_ner']
            sner2 = ['[CLS]']
            shead = d['stanford_head']
            shead2 = ['[CLS]']
            sdep = d['stanford_deprel']
            sdep2 = ['[CLS]']
            for i in range( len(tok_len_map) ):
                n = tok_len_map[i]
                spos2 += [spos[i]]*n
                sner2 += [sner[i]]*n
                shead2 += [shead[i]]*n
                sdep2 += [sdep[i]]*n
            spos2 += ["[SEP]"] * (3 + be1-bs1 + be2-bs2)
            sner2.append( "[SEP]" )
            shead2.append( "[SEP]" )
            sdep2.append( "[SEP]" )
            d['stanford_pos'] = spos2
            d['stanford_ner'] = sner2
            d['stanford_head'] = shead2
            d['stanford_deprel'] = sdep2

    return data



def load_tokens( filename ):
    with open( filename ) as infile:
        data = json.load( infile )
        tokens = []
        for d in data:
            tokens += d['token']
    print( "{} tokens from {} examples loaded from {}.".format( len(tokens), len(data), filename ) )
    return tokens


data = [train, dev, test]
data_name = ["train.json", "dev.json", "test.json"]

if __name__ == "__main__":
    for i, datafile in enumerate(data):
        with open( datafile ) as file:
            indata = json.load( file )
        resultdata = transform( indata, kVocabFile )
        path = './dataset/tacred/' + data_name[i]
        with open(path, "w") as outfile:
            json.dump( resultdata, outfile )
