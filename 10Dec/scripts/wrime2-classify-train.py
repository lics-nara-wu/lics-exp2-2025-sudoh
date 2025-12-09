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

### コマンドライン引数・オプションのチェック
### argparse.ArgumentParserを使う
### ドキュメント https://docs.python.org/ja/3.6/howto/argparse.html
parser = ArgumentParser()
#####
##### 【ここで必要な処理】
##### parser.add_argument()
##### を使って「入力するJSONのファイル名」「出力するモデルのファイル名」を指定する
#####
args = parser.parse_args()

### モデルファイルが既に存在する場合はエラーとしてRuntimeError例外を送出
if os.path.exists(args._): ##### args._ には add_argument で指定した名前を入れる
    raise RuntimeError (f"ERROR: {args._} already exists.") ##### args._ には add_argument で指定した名前を入れる

### JSON形式で記録されたファイルを読み出す（train/dev/testすべてが入っていることに注意）
with open(args._, 'r') as rh: ##### args._ には add_argument で指定した名前を入れる
    DATASET = json.load(rh)

##### データを抽出する関数
def extract_wrime2 (dataset, key='train'):
    """ JSON形式に変換された WRIME ver.2 のデータから文とラベルのリストを抽出する
        dataset['train'] には 学習データ
        dataset['valid'] には 検証データ
        dataset['test'] には 評価データ
        がそれぞれ格納されていて、その中身は辞書形式の変数のリストになっている。
        
        リストに格納された各変数 (data とする) には、
        data["Sentence"] には 文（文字列）
        data["Writer_Sentiment"] には 感情のポジティブ・ネガティブを表す数値 (2, 1, 0, -1, -2)
        がそれぞれ格納されている。
        
        これらの情報を取り出して、
        ・文を _X_str というリストに、
        ・感情の数値を 正・ゼロ・負 の3値にして _y というリストに
        それぞれ格納して返すこと
    """
    _X_str = []
    _y = []

    if key not in dataset:
        raise RuntimeError ("ERROR: {key} is not found in the dataset.")
    else:
        for data in tqdm(dataset[key]):
            ##### 
            ##### 【ここで必要な処理】
            ##### dataという変数から文と感情の数値の情報を取り出し、_X_str と _y に格納する
            ##### _X_str = ["文1", "文2", ...]
            ##### _y = [ラベル1, ラベル2, ...]
            #####

    return _X_str, _y

X_train_str, y_train = extract_wrime2(DATASET)

##### 分類モデルの定義、とりあえず今回はコレでやってみる
##### CountVectorizer は文を空白で区切って単語の列とし、各単語を特徴量として抽出するもの
##### LinearSVC は分類器の一種（線形サポートベクトル分類器）
model_pipeline = make_pipeline(
        CountVectorizer(),
        LinearSVC(random_state=42, max_iter=3000)
        )

model_pipeline.fit(X_train_str, y_train)

### 指定したファイル名にモデルを書き出す
with open(args._, 'wb') as wh: ##### args._ には add_argument で指定した名前を入れる
    pickle.dump(model_pipeline, wh)
    print (f"Success! The trained model has been saved into {args._}.", file=sys.stderr) ##### args._ には add_argument で指定した名前を入れる
