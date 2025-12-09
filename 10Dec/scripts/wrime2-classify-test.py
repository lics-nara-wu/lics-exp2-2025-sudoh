#!/usr/bin/env python3

import sys
import os
import json
import pickle
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from argparse import ArgumentParser

parser = ArgumentParser()
#####
##### parser.add_argument を使って入力ファイル・学習済みモデルファイルの指定を行う
#####
args = parser.parse_args()


with open(args._, 'r') as rh: ##### add_argumentでつけた名前
    DATASET = json.load(rh)

with open(args._, 'rb') as rh: ##### add_argumentでつけた名前
    model_pipeline = pickle.load(rh)

def extract_wrime2 (dataset, key='test'):
    _X_str = []
    _y = []

    if key not in dataset:
        raise RuntimeError ("ERROR: {key} is not found in the dataset.")
    else:
        ##### 学習プログラムの extract_wrime2 と同じにする

        return _X_str, _y


X_test_str, y_test = extract_wrime2(DATASET)
y_predict = model_pipeline.predict(X_test_str)

for _y in y_predict:
    print (_y)
