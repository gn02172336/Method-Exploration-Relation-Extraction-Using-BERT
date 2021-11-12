Method Exploration Relation Extraction Using BERT
=========================

This repo contains the implementation of my [paper](https://github.com/poch4319/Method-Exploration-Relation-Extraction-Using-BERT/blob/main/RD_report_pochun_chang.pdf).


The code is modified from [this repo](https://github.com/elipugh/tacred-scibert-relext). As I can't successfully run the original code, I have applied significant amount of modifications. This repo is suitable for those who want to explore `BERT`'s application for `Relation Extraction`.

**The TACRED dataset**: Details on the TAC Relation Extraction Dataset can be found on [this dataset website](https://nlp.stanford.edu/projects/tacred/).

## Requirements

- Python 3 (tested on 3.6.2)
- [PyTorch](https://github.com/pytorch/pytorch) (tested on 1.0.0)
- [tqdm](https://github.com/tqdm/tqdm)
- [transformers](https://huggingface.co/transformers/) (please use 3.5.1, newer versions might cause error)
- Maybe a couple others
- unzip, wget (for downloading only)

## Preparation

First, download and unzip [12/768 BERT-Base model](https://github.com/google-research/bert/), with:
```
chmod +x download.sh; ./download.sh
```

TACRED dataset is not provided in this repository, you can download it from [Stanford's TACRED Page](https://nlp.stanford.edu/projects/tacred/). Once you have downloaded the dataset, you should tokenize the data first so it is fitting to BERT's input format.

To tokenize the data, first add the path to TACRED dataset you just downloaded in `./data/data_tok.py` between line 12-15.

```
train = './YOUR-TACRED-FOLDER/train.json'
dev = './YOUR-TACRED-FOLDER/dev.json'
test = './YOUR-TACRED-FOLDER/test.json'
```

Then, run the tokenization with:
```
python data/data_tok.py
```


## Training

To train please use the below command. I modified the original code from using `bert-as-service` package to using bert in `huggingface transformers` package. This prevents the complicated use of opening two terminals for `bert-as-service`, but it will require a decent GPU to run a sensible number for batch size. If you just want to test the code, set batch size to 2 will allow you to run the code in your own computer. Else, GPU on google Colab is enough for you to conduct experiments with proper batch size. For how to use this repository on Colab, check `colab.ipynb`.

Model checkpoints and logs will be saved to `./saved_models/00`.

Example Training Command:
```
python train.py --save_epoch 10 --num_epoch 35 --batch_size 20 --special_token --lr 0.03
```


## Evaluation

Run evaluation on the test set with:
```
python eval.py saved_models/00 --dataset test
```

This will use the `best_model.pt` by default. Use `--model checkpoint_epoch_10.pt` to specify a model checkpoint file. Add `--out saved_models/out/test1.pkl` to write model probability output to files.  


## License

All work contained in this package is licensed under the Apache License, Version 2.0. See the included LICENSE file.
