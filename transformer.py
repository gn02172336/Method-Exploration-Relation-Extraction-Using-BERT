import torch
from transformers import *
import json
from bert_serving.client import BertClient

bc = BertClient()


path = './dataset/tacred/train_shuffle.json'
with open(path, 'r') as f:
    datas = json.load(f)

class Bertlayer:
    def __init__(self, model_path, device, opt):
        
        #model_config = transformers.BertConfig.from_pretrained(model_path)
        #model_config.output_hidden_states = True

        #self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)
        #self.model = self.model_class.from_pretrained(model_path, ).to(self.device)
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = BertModel.from_pretrained('bert-base-cased', output_hidden_states = True)
        #self.optimizer = AdamW(self.model.parameters(), lr = 2e-5, eps = 1e-8)
        #if opt['cuda']:
            #self.model.cuda()
                # args.learning_rate - default is 5e-5, our notebook had 2e-5
                #  # args.adam_epsilon  - default is 1e-8.

        # # Whether the model returns all hidden-states.
        # if fix_embeddings:
           # for name, param in self.model.named_parameters():
               # if name.startswith('embeddings'):
                   # param.requires_grad = False

    def get(self, input_batch_list):
        batch_size = len(input_batch_list) 
        words = [sent for sent in input_batch_list]
        word_seq_lengths = torch.LongTensor(list(map(len, words)))
        # 每句句子的长度,获得最长长度
        max_word_seq_len = word_seq_lengths.max().item()
        #word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
        # 长度从长到短排列，并获得由原始排列到从长到短排列的转换顺序 eg:[2,3,1]句子长度，则转换顺序为[1,0,2] #等等會放回去 why???
        batch_tokens = []
        batch_token_ids = []

        #subword_word_indicator = torch.zeros((batch_size, max_word_seq_len), dtype=torch.int64) # ???
        for idx in range(batch_size):
            word_tokens =[] #欠山
            #one_sent_token = input_batch_list[idx]
            #one_subword_word_indicator = []
            #for word in input_batch_list[idx]: #idx 出該句 從該句iterate 每個字
            word_tokens = self.tokenizer.tokenize(input_batch_list[idx]) #不需要 # 得到一句 tokenized
                # 按照wordpiece分词
                #one_subword_word_indicator.append(len(one_sent_token) + 1) # [詞一1, 詞二11] #新句子 會重洗
                # 由于分词之后，和输入的句子长度不同，因此需要解决这个问题，这里保存原始句子中词和分词之后的首个词的对应关系
                #one_sent_token += word_tokens 
                # 针对一句句子，获得分词后的结果
            # 添加 [cls] and [sep] tokens
            word_tokens = ['[CLS]'] + word_tokens + ['[SEP]']  #已經加了
            one_sent_token_id = self.tokenizer.convert_tokens_to_ids(word_tokens) #已經tokenized了
            # token转换id
            print(one_sent_token_id)
            batch_tokens.append(word_tokens)
            batch_token_ids.append(one_sent_token_id)

            #one_subword_word_indicator 為一個list 每個單位就代表 每個來源字 tokenized 後的結尾在句裡的位置

            #subword_word_indicator 為一個array batchsize * 最長的原句長度
            #然後把 one_subword_word_indicator 裝入
            # 只有最長的那句會被塞滿
            # 不夠長的尾部會是0
            #subword_word_indicator[idx, :len(one_subword_word_indicator)] = torch.LongTensor(one_subword_word_indicator)

        token_seq_lengths = torch.LongTensor(list(map(len, batch_tokens))) 
        # token_seq_lengths = tokenized後的 每個句子的長度
        max_token_seq_len = token_seq_lengths.max().item()
        # 计算分词之后最长的句子长度 .max() 會取得最大的元素 但data type 是 torch.tensor  所以用 .item() 製作成 python single value
        batch_token_ids_padded = []
        for the_ids in batch_token_ids:
            batch_token_ids_padded.append(the_ids + [0] * (max_token_seq_len - len(the_ids)))
            # pad tokenized 之後並轉成 id 的 序列 去符合 batch 內最長的 tokenized sequence
        batch_token_ids_padded_tensor = torch.tensor(batch_token_ids_padded) #  [word_perm_idx].to(self.device)
        #subword_word_indicator = subword_word_indicator[word_perm_idx].to(self.device)
        # 都按照之前得出的转换顺序改变为没有分词之前的句子从长到短的排列。
        with torch.no_grad():
            last_hidden_states, _, hidden_states = self.model(batch_token_ids_padded_tensor)
            # BertModel
            # last_hidden_state, pooler_output [CLS], hidden_states opt, attentions opt, cross_attentions opt
        # 提取bert词向量的输出
        
        w_emb = hidden_states[-2] #取倒數第二層

        #Tuple of torch.FloatTensor (one for the output of the embeddings + one for the output of each layer) 共13個item 第一是word emb 後12個是各層輸出

        #batch_word_mask_tensor_list = [] # 可以用來做MASKING ENTITY
        #for idx in range(batch_size):
            #sentence_last_hidden = last_hidden_states[idx]
            #sentence_subword_word_indicator = subword_word_indicator[idx]
            #print(sentence_last_hidden.size())
            #print(sentence_subword_word_indicator)
            #one_sentence_vector = torch.index_select(sentence_last_hidden, 0, sentence_subword_word_indicator)
            #print(one_sentence_vector)
            #one_sentence_vector = one_sentence_vector.unsqueeze(0)
            #print(one_sentence_vector)
            # 根据对应关系，用分词之后的第一个分词来代表整个词，并添加batch的维度
            #batch_word_mask_tensor_list.append(one_sentence_vector)
            
            #sentence_emb = hidden_states[idx] #(batch_size, sequence_length, hidden_size).

        #batch_word_mask_tensor = torch.cat(batch_word_mask_tensor_list, 0)
        #return batch_word_mask_tensor
        return w_emb
        
    def save_model(self, path):
        # 将网上下载的模型文件保存到path中
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)

if __name__ == '__main__':
    sent1 = datas[0]['token']
    sent2 = datas[1]['token']
    sent3 = datas[2]['token']
    input_test_list = ['he is stupid', 'jon eats dog', 'i like to dance and play with girls']
    bert_embedding = Bertvec("C:/Users/abcd8/RD projet/tacred-scibert-relext-master/dataset/cased_L-12_H-768_A-12/", 'cpu', True)
    batch_features = bert_embedding.extract_features(input_test_list)
    #print(batch_features)
    w = bc.encode(input_test_list)
    w = torch.FloatTensor(w)
    print(batch_features.size())
    print(w.size())
    similarity = torch.cosine_similarity(batch_features, w, dim=2)
    print(similarity)