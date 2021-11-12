#!/bin/bash

cd dataset;

echo "==> Downloading BERT..."
wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip

echo "==> Unzipping BERT..."
unzip cased_L-12_H-768_A-12.zip
rm cased_L-12_H-768_A-12.zip

cd ..

echo "==> Done."
