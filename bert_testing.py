from transformers import BertModel, BertTokenizer
import torch


import json

datapath = "data/json/train.json" #/dev.json /test.json
goldpath = "data/gold/train.gold" #/dev.gold /test.gold

DATASET_DIR = "./dataset/tacred/train_mod.json"

with open(DATASET_DIR) as f:
    examples = json.load(f)

print(examples['token'][:2])






text = "Here is the sentence I want embeddings for."
text = "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."
marked_text = "[CLS] " + text + " [SEP]"

tokenizer = BertTokenizer.from_pretrained('bert-base-cased') # follow 原論文使用 cased model


print (marked_text)

tokenized_text = tokenizer.tokenize(marked_text)
print (tokenized_text)

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

segments_ids = [1] * len(tokenized_text)
print (segments_ids)
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-cased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

# Run the text through BERT, and collect all of the hidden states produced
# from all 12 layers. 
with torch.no_grad():
    outputs = model(tokens_tensor, segments_tensors)
# Evaluating the model will return a different number of objects based on 
    # how it's  configured in the `from_pretrained` call earlier. In this case, 
    # becase we set `output_hidden_states = True`, the third item will be the 
    # hidden states from all layers. See the documentation for more details:
    # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
    hidden_states = outputs[2]

print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
layer_i = 0
print ("Number of batches:", len(hidden_states[layer_i]))
batch_i = 0
print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
token_i = 0
print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))


#choice of layers
#summing last four, average all, cat last four

# `hidden_states` is a Python list.
print('      Type of hidden_states: ', type(hidden_states))

# Each layer in the list is a torch tensor
print('Tensor shape for each layer: ', hidden_states[0].size())
token_embeddings = torch.stack(hidden_states, dim=0) #轉list成tensor
# Remove dimension 1, the "batches".
token_embeddings = torch.squeeze(token_embeddings, dim=1)
# Swap dimensions 0 and 1.
token_embeddings = token_embeddings.permute(1,0,2)



def cat_4(token_embeddings):
    # Stores the token vectors, with shape [22 x 3,072]
    token_vecs_cat = []
# `token_embeddings` is a [22 x 12 x 768] tensor.
# For each token in the sentence...
    for token in token_embeddings:
    # `token` is a [12 x 768] tensor
    # Concatenate the vectors (that is, append them together) from the last 
    # four layers.
    # Each layer vector is 768 values, so `cat_vec` is length 3,072.
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
    # Use `cat_vec` to represent `token`.
        token_vecs_cat.append(cat_vec)
    print ('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))
    return token_vecs_cat

def sum_4(token_embeddings):
    # Stores the token vectors, with shape [tokens x 768]
    token_vecs_sum = []
    # `token_embeddings` is a [tokens x 12 x 768] tensor.
    for token in token_embeddings:
    # `token` is a [12 x 768] tensor
    # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)
    # Use `sum_vec` to represent `token`.
        token_vecs_sum.append(sum_vec)
    print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))
    return token_vecs_sum

def sentence(hidden_states):
    # `hidden_states` has shape [13 x 1 x 22 x 768]
# `token_vecs` is a tensor with shape [22 x 768]
    token_vecs = hidden_states[-2][0]
# Calculate the average of all 22 token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)
    return sentence_embedding

final = cat_4(token_embeddings)
final = torch.stack(final, dim=0)
print(final.size)

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