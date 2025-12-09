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
from mylib_wordseg2 import extract_features_from_labeled


### コマンドライン引数とオプションの設定
parser = ArgumentParser()
parser.add_argument("-d", "--debug", action="store_true", help="debug mode")
parser.add_argument("-m", "--model", type=str, required=True, help="model file")
parser.add_argument("traindata", type=str, help="training data file")
args = parser.parse_args()

### モデルファイルが既に存在する場合はエラーとしてRuntimeError例外を送出
if os.path.exists(args.model):
    raise RuntimeError (f"ERROR: {args.model} already exists.")


### 学習データを格納するリスト
X_train_str = [] ### 特徴量のリスト
y_train = [] ### ラベルのリスト

### 学習データ（ラベルが付されたもの）のファイルを開く
with open(args.traindata, 'rt') as rh:
    for line in tqdm(rh):
        line = line.rstrip("\n") ### 行末の改行文字を消しておく
        _X_str_list, _y_list = extract_features_from_labeled(line)

        X_train_str += _X_str_list
        y_train += _y_list


### 学習サンプル数の確認
print (f"INFO: {len(y_train)} training examples are found.", file=sys.stderr)

### 分類モデルの定義
model_pipeline = make_pipeline(
        CountVectorizer(min_df=2),
        LinearSVC(random_state=42)
        )

### 分類モデルの学習
model_pipeline.fit(X_train_str, y_train)

### 学習した分類モデルの保存
with open(args.model, 'wb') as wh:
    pickle.dump(model_pipeline, wh)
    print (f"Success! The trained model has been saved into {args.model}.", file=sys.stderr)
