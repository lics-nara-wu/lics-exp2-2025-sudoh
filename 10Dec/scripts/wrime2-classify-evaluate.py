#!/usr/bin/env python3

import sys
import os
import json
import pickle
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from argparse import ArgumentParser

parser = ArgumentParser()
##### parser.add_argument でWRIME2のデータと予測結果のファイルをそれぞれ指定
args = parser.parse_args()


with open(args._, 'r') as rh: ##### add_argumentでつけた名前
    DATASET = json.load(rh)

y_predict = []
with open(args._, 'r') as rh: ##### add_argumentでつけた名前
    for line in rh:
        ##### 出力された予測ラベルを y_predict に格納する
        ##### ファイルに出力されたラベルは文字列型になっているので、int()で整数型に変換して格納すること

def extract_wrime2 (dataset, key='test'):
    _X_str = []
    _y = []

    if key not in dataset:
        raise RuntimeError ("ERROR: {key} is not found in the dataset.")
    else:
        ##### 学習用プログラムと同じ

        return _X_str, _y


X_str_test, y_test = extract_wrime2(DATASET)
test_accuracy = accuracy_score(y_test, y_predict)
print (f"Test accuracy: {100*test_accuracy:.3g} %")
