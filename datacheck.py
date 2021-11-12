import json

import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from pandas import DataFrame 
from transformers import *

def readtrain():
    path = './dataset/tacred/train.json'

    with open(path, 'r') as f:
        datas = json.load(f)

    return datas

def readtrain2():
    path2 = './old dataset/og tacred/train_origin.json'
    with open(path2, 'r') as f2:
        datas2 = json.load(f2)
    return datas2

def inner(d):
    sstart = d['subj_start']
    send = d['subj_end']
    ostart = d['obj_start']
    oend = d['obj_end']
    sentece = d['token']
    stype = d['subj_type']
    otype = d['obj_type']
    relation = d['relation']
    print(sentece)
    print(sstart, send)
    print(ostart, oend)
    print(sentece[sstart:send])
    print(sentece[ostart:oend])
    print(stype)
    print(otype)
    print(sentece)
    print(relation)


# 
#
def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def checkpos(d):
    tokens = d['token']
    l = len(tokens)
    subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
    obj_positions = get_positions(d['obj_start'], d['obj_end'], l)
    print(tokens)
    print(subj_positions)
    print(obj_positions)

def maxlen(datas):
    maxlens = 0
    for d in datas:
        tokens = d['token']
        if len(tokens) > maxlens:
            maxlens = len(tokens)
    print(maxlens)

#maxlen(datas)

''' parsing log
with open('./log1.txt', 'r') as files:
    lines = files.readlines()
    doc = dict()
    doc['epoch'] = []
    doc['train_loss'] = []
    doc['dev_loss'] = []
    doc['dev_f1'] = []
    doc['dev_precesion'] = []
    doc['dev_recall'] = []
    for line in lines[1:]:
        line = line.strip()
        line = line.split('\t')
        doc['epoch'].append(float(line[0]))
        doc['train_loss'].append(float(line[1]))
        doc['dev_loss'].append(float(line[2]))
        doc['dev_f1'].append(float(line[3]))
        doc['dev_precesion'].append(float(line[4]))
        doc['dev_recall'].append(float(line[5]))

doc['dev_precesion'][0] = 0
df = DataFrame(doc)
df.plot(x ='epoch', y=['train_loss', 'dev_loss'], kind = 'line')
plt.show()

df.plot(x ='epoch', y=['dev_f1', 'dev_precesion', 'dev_recall'], kind = 'line')
plt.show()
''' 

"""

'''
['id', 
'docid',
 'relation', 
 'token',
  'subj_start',
   'subj_end', 
   'obj_start',
    'obj_end',
     'subj_type', 
     'obj_type',
      'stanford_pos', 
      'stanford_ner', 
      'stanford_head', 
      'stanford_deprel', 
      'token_drop'])
"""

''' concatenate
a = torch.zeros([50, 86, 768], dtype=torch.int32)
b = torch.ones([50, 40], dtype=torch.int32)
print(a.size())

print(b.size())
c = torch.cat((a, b), 1)
print(c.size())
'''

def tok(d):
    mapping = { '-LRB-': '(',
                '-RRB-': ')',
                '-LSB-': '[',
                '-RSB-': ']',
                '-LCB-': '{',
                '-RCB-': '}',
                    0: '[UNK]',
                    1: '[PAD]' }
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
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
        token = str(token)

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

    d['token'] = bert_tokens

    return d

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx) + \
            list(range(1, length-end_idx+1))

''' test data preprocessing, need to change path in read train
datas = readtrain()
d = datas[15]
inner(d)
d = tok(d)
inner(d)
a = get_positions(d['subj_start'], d['subj_end'], len(d['token']))
b = get_positions(d['obj_start'], d['obj_end'], len(d['token']))
print(a)
print(b)
print(len(d['token']))
print(len(a),len(b))
'''
old = readtrain2()
datas = readtrain()
oldd = old[82]
d = datas[82]

inner(oldd)
inner(d)