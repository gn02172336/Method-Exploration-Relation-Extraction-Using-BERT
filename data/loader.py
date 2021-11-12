"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np
from tqdm import tqdm
from transformers import *

from utils import constant, helper, vocab

#from bert_serving.client import BertClient



class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab=None, life=None, evaluation=False):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        #self.bc = BertClient()
        self.life = life

        with open(filename) as infile:
            data = json.load(infile)
        if opt['bert']:
            data = self.preprocess_bert(data, vocab, opt)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            if opt['special_token']:
                ############################ part for create speical token
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
                special_tok = []
                obj = 19
                subj = 4
                for num in range(obj):
                    special_tok.append("[OBJ" + str(num) + "]")
                for num in range(subj):
                    special_tok.append("[SUBJ" + str(num) + "]")
                special_tokens_dict = {'additional_special_tokens': special_tok}
                num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
                ############################ part for create speical token
        else:
            data = self.preprocess(data, vocab, opt)
        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]

        if self.life:
            id2label = dict([(v,k) for k,v in constant.LIFE_LABEL_TO_ID.items()])
        if not self.life:
            id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
        self.labels = [id2label[d[-1]] for d in data] 
        self.num_examples = len(data)
        
        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data #data has been devided with batch size into serveral batches
        print("{} batches created for {}".format(len(data), filename))
        print('batch count', len(data))


    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in data:
            tokens = d['token']
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se+1] = ['SUBJ-'+d['subj_type']] * (se-ss+1)
            tokens[os:oe+1] = ['OBJ-'+d['obj_type']] * (oe-os+1)
            tokens = map_to_ids(tokens, vocab.word2id)
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            l = len(tokens)
            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            obj_positions = get_positions(d['obj_start'], d['obj_end'], l)
            if not self.life:
                relation = constant.LABEL_TO_ID[d['relation']]
            if self.life:
                relation = constant.LIFE_LABEL_TO_ID[d['relation']]

            processed += [(tokens, pos, ner, deprel, subj_positions, obj_positions, relation)]
        return processed

    def preprocess_bert(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in tqdm(data):
            tokens = d['token']
            for i, token in enumerate(tokens):
                if type(token) != str:
                    tokens[i] = str(token)
            # anonymize tokens [skip, doesn't prevent overfitting?]
            # ss, se = d['subj_start'], d['subj_end']
            # os, oe = d['obj_start'], d['obj_end']
            # TODO check indexing
            # tokens[ss:se] = ['SUBJ-'+d['subj_type']] * (se-ss)
            # tokens[os:oe] = ['OBJ-'+d['obj_type']] * (oe-os)

            # tokens = map_to_ids(tokens, vocab.word2id)

            # pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            # ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            # deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            
            l = len(tokens)
            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            obj_positions = get_positions(d['obj_start'], d['obj_end'], l)
            assert l == len(subj_positions)
            assert l == len(obj_positions)

            if self.life:
                relation = constant.LIFE_LABEL_TO_ID[d['relation']]
            if not self.life:
                relation = constant.LABEL_TO_ID[d['relation']]
            processed += [(tokens, None, None, None, subj_positions, obj_positions, relation)]
        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        #return 50
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]  #batch number, used in train.py iteration of batch data, here we get the actual batch based on batch number
        batch_size = len(batch)
        batch = list(zip(*batch))  #batch = [(tokens, pos, ner, deprel, subj_positions, obj_positions, relation), ... , ... ,...]
        # *batch = (tokens, pos ....) , (tokens, pos...) no more list 
        # zip makes each element in the same position together -> (tokens, tokens.....) (pos, pos ......)
        # list turn them into list [(all tokens), (all pos), (all ner), ...]
        assert len(batch) == 7 #7 elements

        # sort all fields by lens for ,easy RNN operations
        lens = [len(x) for x in batch[0]] #batch[0] is (tokens) here lens document all the length of token list in this batch
        #batch, orig_idx = sort_all(batch,lens) 
        orig_idx = None

        # word dropout
        # 刪掉dropout 好了 注意dropout 是給 dropout id 而不是 [UNK] 超怪
        #if not self.eval:
            #words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        #else:
        words = batch[0] # get the (all tokens)
        if self.opt['bert']:
            words = processforbert(words, self.tokenizer)
            pos, ner, deprel = None, None, None

        else:
            # convert to tensors
            words = get_long_tensor(words, batch_size)
            pos = get_long_tensor(batch[1], batch_size)
            ner = get_long_tensor(batch[2], batch_size)
            deprel = get_long_tensor(batch[3], batch_size)
        masks = torch.eq(words, 0)
        subj_positions = get_long_tensor(batch[4], batch_size)  #get_long_tensor 會幫忙pad 所以要在此之前刪掉sep後面的東西 但是 word embedding 部分要在Rnn那邊再刪 並重新確認長度
        obj_positions = get_long_tensor(batch[5], batch_size)

        rels = torch.LongTensor(batch[6])
        #if self.switch == False and self.eval == False:
            #if all(i == True for i in self.bert_encode):
                #self.complete_bert()

        return (words, masks, pos, ner, deprel, subj_positions, obj_positions, rels, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def complete_bert(self): #yet finish
        self.switch = True
        savedata = dict()
        for i in range(self.__len__()):
            savedata[i] = self.__getitem__(i)
        path = './dataset/tacred/everything.json'
        with open(path, "w") as outfile:
            json.dump( savedata, outfile )
            
        

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx) + \
            list(range(1, length-end_idx+1))

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def padded(tokens_list):
    result = []
    token_len = max(len(x) for x in tokens_list)
    for sentence in tokens_list:
        while len(sentence) < token_len:
            sentence.append('[PAD]')
        result.append(sentence)
    return result

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [str(constant.UNK_ID) if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]


def processforbert(setences, tokenizer):
        tokens_ids = []
    
        #subword_word_indicator = torch.zeros((batch_size, max_word_seq_len), dtype=torch.int64) # ???
        for tokens in setences:
            one_sent_token_id = tokenizer.convert_tokens_to_ids(tokens) #已經tokenized了
            tokens_ids.append(one_sent_token_id)
            
        sentence_length = torch.LongTensor(list(map(len, setences))) 
        max_sentence_len = sentence_length.max().item()
        # 计算分词之后最长的句子长度 .max() 會取得最大的元素 但data type 是 torch.tensor  所以用 .item() 製作成 python single value
        tokens_ids_padded = []
        for the_ids in tokens_ids:
            tokens_ids_padded.append(the_ids + [0] * (max_sentence_len - len(the_ids)))
        tokens_ids_padded_tensor = torch.tensor(tokens_ids_padded) 

        return tokens_ids_padded_tensor

        