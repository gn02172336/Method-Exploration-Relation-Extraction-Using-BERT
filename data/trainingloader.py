"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np
from tqdm import tqdm

from utils import constant, helper, vocab

from bert_serving.client import BertClient

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab=None, life=None, evaluation=False):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.bc = BertClient()
        self.life = life

        self.batch_num = 0
        with open(filename) as infile:
            data = json.load(infile)
        
        data = self.preprocess_bert(data, vocab, opt)


        id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
        self.labels = [id2label[d[-1]] for d in data] 
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data #data has been devided with batch size into serveral batches
        print("{} batches created for {}".format(len(data), filename))
        print('batch count', len(data))

  

    def preprocess_bert(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in tqdm(data):
            tokens = d['token_drop']
            for i, tok in enumerate(tokens):
                if tok == '1':
                    tokens[i] = '[UNK]'
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

        orig_idx = None

        if key == 0:
            self.loaded = torch.load('embedding0')
        elif (key + 99) % 100 == 0 and key < 1301:
            self.loaded = torch.load('embedding' + str(key+99))
        if key == 1301:
            self.loaded = torch.load('embedding1301-1363')
    
        w = self.loaded[key]

        
        if self.opt['bert']: 
            words = torch.FloatTensor(w)
            pos, ner, deprel = None, None, None

        else:
            # convert to tensors
            words = get_long_tensor(words, batch_size)

            pos = get_long_tensor(batch[1], batch_size)
            ner = get_long_tensor(batch[2], batch_size)
            deprel = get_long_tensor(batch[3], batch_size)
        gi = 0
        sent = batch[0]
        sent = padded(sent)
        for s in sent:
            if len(s) > gi:
                gi = len(s)
        print(gi)
        wordss = self.bc.encode(sent, is_tokenized=True)
        wordss =  torch.FloatTensor(wordss)
        print(wordss.size())
        masks = torch.eq(words, 0)
        subj_positions = get_long_tensor(batch[4], batch_size)
        print(subj_positions.size())
        obj_positions = get_long_tensor(batch[5], batch_size)
        print(obj_positions.size())

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
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def padded(tokens_list): #改成 BERT 的 [PAD] 不是 <PAD>
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

def word_dropout(tokens, dropout):  #bad 隨機取代句子裡的token 成 字串 1  跟 bert 的 [UNK] 不搭
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [str(constant.UNK_ID) if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]

def embedding_extraction(data):
    for d in tqdm(data):
        tokens = d['token']
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        model = BertModel.from_pretrained('bert-base-cased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
        model.eval()
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0) #轉list成tensor
        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1,0,2)
        final = cat_4(token_embeddings)
        final = torch.stack(final, dim=0)   
