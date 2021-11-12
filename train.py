"""
Train a model on TACRED.
"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import os
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
import json

from data.loader import DataLoader
#from data.trainingloader import DataLoader as TrainLoader
#from data.devloader import DataLoader as DevLoader
from model.rnn import RelationModel
from utils import scorer, constant, helper
from utils.vocab import Vocab

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset/tacred') # not needed actually, ignore
parser.add_argument('--vocab_dir', type=str, default='dataset/vocab')
parser.add_argument('--train_file', type=str, default='train.json')
parser.add_argument('--dev_file', type=str, default='dev.json')
parser.add_argument('--emb_dim', type=int, default=768, help='Word embedding dimension.')
parser.add_argument('--ner_dim', type=int, default=0, help='NER embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=0, help='POS embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=768, help='RNN hidden state size.') #200
parser.add_argument('--mlp_dim', type=int, default=300, help='Word embedding dimension.') # doesnt exist
parser.add_argument('--num_layers', type=int, default=1, help='Num of RNN layers.') #2
parser.add_argument('--dropout', type=float, default=0.5, help='Input and RNN dropout rate.')
parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N embeddings.')
parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
parser.add_argument('--no-lower', dest='lower', action='store_false')
parser.set_defaults(lower=False)

parser.add_argument('--no-attn', dest='attn', action='store_false')
parser.set_defaults(attn=False)
parser.add_argument('--pe_dim', type=int, default=20, help='Position encoding dimension.') # 20

parser.add_argument('--lr', type=float, default=0.05, help='Applies to SGD and Adagrad.') # 1 0.00005
parser.add_argument('--lr_decay', type=float, default=0.9)
parser.add_argument('--optim', type=str, default='sgd', help='sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=45) #30
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=5, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
parser.add_argument('--bert', type=bool, default=True, help='Use Bert.')
parser.add_argument('--special_token', action='store_true', help='Whether bert input has added token')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# make opt
opt = vars(args)
opt['num_class'] = len(constant.LABEL_TO_ID)

# load vocab only if bert is not on
vocab = None
bert_con = './dataset/cased_L-12_H-768_A-12/bert_config.json'
with open( bert_con ) as file:
    content = json.load( file )
opt['vocab_size'] = content["vocab_size"]

# load data
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
train_batch = DataLoader(opt['data_dir'] + "/"+opt['train_file'], opt['batch_size'], opt, evaluation=False)
dev_batch = DataLoader(opt['data_dir'] + "/"+opt['dev_file'], opt['batch_size'], opt, evaluation=True)

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)

# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)

file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\tdev_loss\tdev_f1\tPrecision\tRecall")

# print model info
helper.print_config(opt)

# model
model = RelationModel(opt)


id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
dev_f1_history = []
current_lr = opt['lr']

global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
#print (train_batch[500]) #checking the batch content
max_steps = len(train_batch) * opt['num_epoch']

# start training
for epoch in range(1, opt['num_epoch']+1):  # epoch default 30 batch 50
    train_loss = 0
    for i, batch in enumerate(train_batch):
        start_time = time.time()
        global_step += 1 # one batch takes one step to run.  data_size / batch_size = how many batch in one epoch. totol_step = data_size * epoch / batch_size
        loss = model.update(batch)  #model is updated every batch, so if the batch is big, model updated less and takes longer for each update
        train_loss += loss
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            print(format_str.format(datetime.now(), global_step, max_steps, epoch,\
                    opt['num_epoch'], loss, duration, current_lr))

    # eval on dev
    print("Evaluating on dev set...")
    predictions = []
    dev_loss = 0
    for i, batch in enumerate(dev_batch):
        preds, _, loss = model.predict(batch)
        predictions += preds
        dev_loss += loss
    predictions = [id2label[p] for p in predictions]
    dev_p, dev_r, dev_f1 = scorer.score(dev_batch.gold(), predictions, verbose=True)
    
    train_loss = train_loss / train_batch.num_examples * opt['batch_size'] # avg loss per batch
    dev_loss = dev_loss / dev_batch.num_examples * opt['batch_size']
    print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}".format(epoch,\
            train_loss, dev_loss, dev_f1))
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.4f}\t{:.4f}".format(epoch, train_loss, dev_loss, dev_f1,dev_p,dev_r))

    # save
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    model.save(model_file, epoch)
    if epoch == 1 or dev_f1 > max(dev_f1_history):
        copyfile(model_file, model_save_dir + '/best_model.pt')
        print("new best model saved.")
    if epoch % opt['save_epoch'] != 0:
        os.remove(model_file)
    
    # lr schedule
    if len(dev_f1_history) > 10 and dev_f1 <= dev_f1_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad']:
        current_lr *= opt['lr_decay']
        model.update_lr(current_lr)

    dev_f1_history += [dev_f1]
    print("")

print("Training ended with {} epochs.".format(epoch))

