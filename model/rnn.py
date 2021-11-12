"""
A rnn model for relation extraction, written in pytorch.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from transformers import *
from utils import constant, torch_utils
from model import layers

class RelationModel(object):
    """ A wrapper class for the training and evaluation of models. """
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.model = PositionAwareRNN(opt, emb_matrix)
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])


    
    def update(self, batch):
        """ Run a step of forward and backward model update. """
        if self.opt['cuda']:
            inputs = [b.cuda() if type(b) is torch.Tensor else None for b in batch[:7]] 
            labels = batch[7].cuda()
        else:
            inputs = [b for b in batch[:7]] 
            labels = batch[7]
        #(words, masks, pos (none), ner(none), deprel(none), subj_positions, obj_positions), rels, orig_idx)

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, _ = self.model(inputs)
        loss = self.criterion(logits, labels)
        
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()
        return loss_val

    def predict(self, batch, unsort=False):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        if self.opt['cuda']:
            inputs = [b.cuda() if type(b) is torch.Tensor else None for b in batch[:7]]
            labels = batch[7].cuda()
        else:
            inputs = [b for b in batch[:7]]
            labels = batch[7]

        #orig_idx = batch[8]

        # forward
        self.model.eval()
        logits, _ = self.model(inputs)
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            print('unsorting!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1')
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs)))]
        return predictions, probs, loss.data.item()

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                'epoch': epoch
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

class PositionAwareRNN(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt, emb_matrix=None):
        super(PositionAwareRNN, self).__init__()
        self.drop = nn.Dropout(opt['dropout'])
        self.opt = opt
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        
        #if opt['cuda']:
         #   self.bertmodel = BertModel.from_pretrained('bert-base-cased', output_hidden_states = True).to('cuda')
        #else:
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.bertmodel = BertModel.from_pretrained('bert-base-cased', output_hidden_states = True).to('cuda')
        if opt["special_token"]:
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
            self.bertmodel.resize_token_embeddings(len(self.tokenizer))

        if opt['pos_dim'] > 0: # default 0
            self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim'],
                    padding_idx=constant.PAD_ID)
        if opt['ner_dim'] > 0: # default 0
            self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim'],
                    padding_idx=constant.PAD_ID)
        
        input_size = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim'] + opt['pe_dim'] + opt['pe_dim'] # default 768, 0, 0
        self.rnn = nn.LSTM(input_size, opt['hidden_dim'], opt['num_layers'], batch_first=True,\
                dropout=opt['dropout'], bidirectional=True)  # input dim = 768, h = 200, layers = 2, dropout = 0.5 default add bidrectional
        self.mlinear = nn.Linear(opt['hidden_dim'], opt['mlp_dim'])  # added!!!!
        self.linear = nn.Linear(opt['mlp_dim'], opt['num_class']) # modified! to categories

        if opt['attn']: 
            self.attn_layer = layers.PositionAwareAttention(opt['hidden_dim'],
                    opt['hidden_dim'], 2*opt['pe_dim'], opt['attn_dim'])
        self.pe_emb = nn.Embedding(constant.MAX_LEN * 2 + 1, opt['pe_dim'], padding_idx=constant.PAD_ID) #change *2 + 1 改成 不改

        self.opt = opt
        self.topn = self.opt.get('topn', 1e10)
        self.use_cuda = opt['cuda']
        self.emb_matrix = emb_matrix
        self.init_weights()
    
    def init_weights(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0, 1.0) # keep padding dimension to be 0
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        if self.opt['pos_dim'] > 0:
            self.pos_emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        if self.opt['ner_dim'] > 0:
            self.ner_emb.weight.data[1:,:].uniform_(-1.0, 1.0)

        self.linear.bias.data.fill_(0)
        init.xavier_uniform_(self.linear.weight, gain=1) # initialize linear layer 

        self.mlinear.bias.data.fill_(0) #added
        init.xavier_uniform_(self.mlinear.weight, gain=1) #added

        #if self.opt['attn']:
        self.pe_emb.weight.data.uniform_(-1.0, 1.0) #修改拿出來

        # decide finetuning
        if self.topn <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif not self.opt['bert'] and self.topn < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.topn))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.topn))
        else:
            print("Finetune all embeddings.")

    def zero_state(self, batch_size): 
        state_shape = (self.opt['num_layers']*2, batch_size, self.opt['hidden_dim']) # first arguement *2
        h0 = c0 = torch.zeros(*state_shape, requires_grad=False)
        if self.use_cuda:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0
    
    def forward(self, inputs):
        words, masks, pos, ner, deprel, subj_pos, obj_pos = inputs # unpack
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        batch_size = words.size()[0]

        #print(words[0,:])
        #print(subj_pos[0,:])
        #print(obj_pos[0,:])
        #print(list(map(len, [words[0,:], subj_pos[0,:], obj_pos[0,:]])))
        # embedding lookup
        if self.opt['bert']:
            #with torch.no_grad():
            last_hidden_states, _, hidden_states = self.bertmodel(words)
            w_emb = hidden_states[-2]

            subj_pe_inputs = self.pe_emb(subj_pos + constant.MAX_LEN) #subj_pos + constant.MAX_LEN 數值至少要1 pad = 0 所以 初始化emb才要 x2 + 1
            obj_pe_inputs = self.pe_emb(obj_pos + constant.MAX_LEN) #change

            inputs = torch.cat((w_emb, subj_pe_inputs, obj_pe_inputs), 2)

        else:
            inputs = [self.emb(words)]
            if self.opt['pos_dim'] > 0:
                inputs += [self.pos_emb(pos)]
            if self.opt['ner_dim'] > 0:
                inputs += [self.ner_emb(ner)]
            # TODO check dropout
            inputs = self.drop(torch.cat(inputs, dim=2)) # add dropout to input

        input_size = inputs.size(2) #batch, maxlen, bertemb

        # rnn
        h0, c0 = self.zero_state(batch_size)
        # inputs = nn.utils.rnn.pack_padded_sequence(inputs, seq_lens, batch_first=True)
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        # outputs, output_lens = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        hidden = self.drop(ht[-1,:,:] + ht[-2,:,:]) # get the outmost layer h_n # changed!!!
        outputs = self.drop(outputs)
        
        # attention
        if self.opt['attn']:
            # convert all negative PE numbers to positive indices
            # e.g., -2 -1 0 1 will be mapped to 98 99 100 101
            subj_pe_inputs = self.pe_emb(subj_pos + constant.MAX_LEN) # subj_pos + constant.MAX_LEN
            obj_pe_inputs = self.pe_emb(obj_pos + constant.MAX_LEN) 
            pe_features = torch.cat((subj_pe_inputs, obj_pe_inputs), dim=2)
            final_hidden = self.attn_layer(outputs, masks, hidden, pe_features)
        else:
            final_hidden = hidden

        logits = self.mlinear(final_hidden) # added
        logits = self.linear(logits) # modified

        return logits, final_hidden
    

