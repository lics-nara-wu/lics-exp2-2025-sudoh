# 第12回 2024-12-25

## はじめに
須藤の担当回では日本語の自然言語処理に関わるプログラムの作成を通じて
- (1) 機械学習によるデータ分類（パターン認識）の基礎
- (2) [scikit-learn](https://github.com/scikit-learn/scikit-learn)を使った機械学習の基礎
- (3) 機械学習による日本語文書分類
- (4) 機械学習による日本語の分かち書き
を学びます。

第11回と第12回では (4) を行います。

## 【再掲】実験環境
- G棟のLinuxを前提とします
- 実装言語はG棟のLinuxで標準利用可能な Python を基本とします
  - 自主的に他の言語で実装することは妨げませんが、scikit-learnと同じようにやるのはおそらくかなり大変です
  - ただし、データの前処理に係る部分はできるだけ提供している Python スクリプトを利用してください

> [!WARNING]
> ※ このPythonはバージョンが古いので、Pythonの最近の機能やライブラリがいろいろ使えません。
> G棟の環境で動くような仕組みで説明しますので、今どきの環境ではそのまま動かないかもしれませんがご容赦ください。

## 【再掲】Pythonのリファレンス
- [Python 3.6ドキュメント](https://docs.python.org/ja/3.6/)
- [Python 3.6標準ライブラリリファレンス](https://docs.python.org/ja/3.6/library/index.html)
- [本実験で使うPythonのライブラリ等の使い方説明](https://github.com/lics-nara-wu/lics-exp2-2024/edit/main/README_python.md)

> [!TIP]
> [Pythonの機能についての説明](https://github.com/lics-nara-wu/lics-exp2-2024/blob/main/README_python.md)にこの実験で使うPythonの機能の説明を記載します。
>
> 質問等で共有が必要になったときには随時更新します。

## 当日の実験の流れ
以下のような流れで進めます。自分で分かるという人はどんどん先に進めていただいてかまいません。

1. [実験の目的と内容](#1-実験の目的と内容)
2. [Pythonの環境設定の呼び出し](#2-Pythonの環境設定の呼び出し)
3. [課題の技術的説明](#3-課題の技術的説明)
4. [必要な関数の作成](#4-必要な関数の作成)
5. [課題提出（時間内に終わらなければ提出期限までに提出すればOK）](#6-課題提出時間内に終わらなければ提出期限までに提出すればok)


## 1. 実験の目的と内容
スライドを使って説明します。スライドはLMSで共有します。

## 2. Pythonの環境設定の呼び出し
[第11回での環境設定](https://github.com/lics-nara-wu/lics-exp2-2024/blob/main/18Dec/README.md)が終了しているものとして、それを呼び出します。
```
EXPDIR=${HOME}/exp2_2024_nlp
cd ${EXPDIR}
source ${EXPDIR}/.venv/bin/activate
```
`exp_2024_nlp` という仮想環境が有効になっていることを確認してください。
```
(exp2_2024_nlp) [sudoh@remote01 exp2_2024_nlp]$
```

環境変数の設定をします。
```
export LANG=ja_JP.utf8
export LC_ALL=ja_JP.utf8
```
また、合わせてターミナルの文字エンコーディングを Unicode > UTF-8 に変更します。

## 3. 課題の技術的説明
スライドを使って説明します。スライドはLMSで共有します。

## 4. 必要な関数の作成
[第11回](https://github.com/lics-nara-wu/lics-exp2-2024/blob/main/18Dec/README.md)で分かち書きする基本的な仕組みを作ったので、それを改造してより良い分かち書きができないかを検討します。
- 組み合わせ特徴量
- 辞書情報に基づく特徴量

この機能を前回作成した `mylib_wordseg.py` をコピーした別ファイル `mylib_wordseg2.py` に定義された関数内に実装してください。まず以下のコマンドでコピーを行います。
```
cp mylib_wordseg.py mylib_wordseg2.py
```

なお、実行するプログラムである以下の三つはテンプレート通りでこれらのファイルについては内容変更不要です（前回から変更があるので名前も変えています）。
- [`wordseg2-train.py`](https://github.com/lics-nara-wu/lics-exp2-2024/blob/main/25Dec/scripts/wordseg2-train.py): 学習プログラム
- [`wordseg2-predict.py`](https://github.com/lics-nara-wu/lics-exp2-2024/blob/main/25Dec/scripts/wordseg2-predict.py): 予測プログラム
- [`wordseg2-evaluate.py`](https://github.com/lics-nara-wu/lics-exp2-2024/blob/main/25Dec/scripts/wordseg2-evaluate.py): 評価プログラム


### 4.1. 特徴量の抽出
`mylib_wordseg2.py` の `extract_features` という関数を自分で編集し、有用そうな特徴量を追加してみてください。

辞書情報を参照できるようにするため、`mylib_wordseg2.py` の冒頭が以下のようになるように pickle 関係の行を書き足してください。
```
import sys
import regex as re

import pickle

with open('/export/home/ics/sudoh/Project/Exp2/2024/data/unidic.pkl', 'rb') as rh:
    DICT = pickle.load(rh)

##### 課題
```

#### 4.1.1. 組み合わせ特徴量
『「私」と「は」』の間は切れやすい、という直観を特徴量として活用したいと考えたとき、前回は
- 「私」という文字がすぐ左にある
- 「は」という文字がすぐ右にある
という特徴量をそれぞれ定義して使っていました。

今回使っている分類器 `linearSVC` はこの二つの特徴量が同時に観測されている、という情報をうまく活用できないので、
- 『「私」という文字がすぐ左にあり』かつ『「は」という文字がすぐ右にある』
というような特徴量を使うことで、「私」と「は」の間は切れやすい、ということをより効果的に学習させることができます。

#### 4.1.2. 辞書情報に基づく特徴量
「奈良」というのは一つの単語であるはずで、その両端は切れやすく、その内部は切れにくい、というような直観が働くと思いますが、
その直観を特徴量として使うためにはそもそもどういう単語が存在するのか、という辞書の情報が必要です。

なお、辞書情報の活用の詳細については本ページ末尾にある論文に具体的な記載がありますので、参考にしてください。

>[!TIP]
>文字のリスト `char_list` に対して 例えばi番目からi+2番目の部分のみを取り出したいときはスライスという機能を使います。
>```
>char_list[i:i+3]
>```
>コロン(:)の後ろのインデックスは必要な範囲より1大きい値になっていなければならないことに注意してください。

>[!TIP]
>辞書の情報は辞書型の変数`DICT`で参照できるようになっています。
>辞書に含まれているかどうかを確認するには以下のようにします。
>```
>if "奈良" in DICT:
>    （"奈良"が辞書に含まれているときの処理）
>```
>なお、`char_list`のスライスで得られるのは文字列ではなく文字のリストなので、以下のようにして文字列に変換する必要があります。
>```
>"".join(char_list[i:i+3])
>```
>参考：https://note.nkmk.me/python-string-concat/


### 4.2. 学習プログラムの実行
特徴量抽出関数ができたら以下のプログラムを実行してモデルを作成してみてください。
```
python3 wordseg2-train.py -m wordseg2.model ${EXPDIR}/data/jawiki-20241201-pages-train-tiny.ja.tok.label
```
無事完了したら `wordseg2.model` というファイルができるはずです。

### 4.3. 予測プログラムの実行
その後、以下のプログラムを実行し、分かち書きができているか確認してみてください。
```
head -n 3 ${EXPDIR}/data/jawiki-20241201-pages-test.ja | python3 wordseg2-predict.py -m wordseg2.model
```

### 4.4. 評価プログラムの実行
その後、以下のプログラムを実行し、テストデータに対する分かち書きを行います。
（あまり高速化の工夫を行っていないので、数分かかります）
```
python3 wordseg2-predict.py -m wordseg2.model < ${EXPDIR}/data/jawiki-20241201-pages-test.ja > test2.txt
```

最後に、以下のプログラムを実行し、精度評価を行ってください。
```
python3 wordseg2-evaluate.py -r ${EXPDIR}/data/jawiki-20241201-pages-test.ja.tok.label test2.txt
```

前回作成したプログラムで実行した場合との精度も比較してみましょう。
```
python3 wordseg2-predict.py -m wordseg.model < ${EXPDIR}/data/jawiki-20241201-pages-test.ja > test.txt
python3 wordseg2-evaluate.py -r ${EXPDIR}/data/jawiki-20241201-pages-test.ja.tok.label test.txt
```

### 4.5. 少し大きな学習データを使った実験
時間があれば、少し大きな学習データである `jawiki-20241201-pages-train-small.ja.tok.label` を使ったモデルも作ってみてください。
```
python3 wordseg2-train.py -m wordseg2.model2 ${EXPDIR}/data/jawiki-20241201-pages-train-small.ja.tok.label
python3 wordseg2-predict.py -m wordseg2.model2 < ${EXPDIR}/data/jawiki-20241201-pages-test.ja > test2S.txt
python3 wordseg2-evaluate.py -r ${EXPDIR}/data/jawiki-20241201-pages-test.ja.tok.label test2S.txt
```

## 5. 課題提出（時間内に終わらなければ提出期限までに提出すればOK）
LMSの「課題（第12回、自然言語処理2）」のところに以下を提出してください。
- 作成したプログラム（`mylib_wordseg2.py`のみ）

> [!IMPORTANT]
> 提出期限は **2025-01-08 (水) 23:59 (日本標準時)** です。
> 
> 提出期限後の提出も受け付けますが、減点対象です。

## 参考論文
Graham Neubig, 中田 陽介, 森 信介,
[点推定と能動学習を用いた自動単語分割器の分野適応（PDF）](https://www.anlp.jp/proceedings/annual_meeting/2010/pdf_dir/C4-3.pdf)
言語処理学会第16回年次大会発表論文集 pp.912-915 (2010)

（4.1節が特徴量抽出に関する情報が記載されている箇所です）
