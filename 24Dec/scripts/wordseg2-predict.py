#!/usr/bin/env python3

import sys
import os
import pickle
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from argparse import ArgumentParser

### 自作ライブラリ (mylib_wordseg.py) を呼び出す
from mylib_wordseg2 import extract_features_from_unlabeled, apply_wordseg


### コマンドライン引数とオプションの設定
parser = ArgumentParser()
parser.add_argument("-d", "--debug", action="store_true", help="debug mode")
parser.add_argument("-m", "--model", type=str, required=True, help="model file")
args = parser.parse_args()

### モデルファイルを読み込む
with open(args.model, 'rb') as rh:
    model_pipeline = pickle.load(rh)


### テストデータ（ラベルが付されていないもの）を標準入力から読み込み、各文字間のラベルを予測する
for line in sys.stdin:
    line = line.rstrip("\n") ### 行末の改行文字を消しておく
    if len(line) > 1:
        X_test_str = extract_features_from_unlabeled(line)
        y_predict = model_pipeline.predict(X_test_str)

        if args.debug: print (y_predict)
        line_segmented = apply_wordseg(line, y_predict)
        print (line_segmented)
    else:
        print (line)
