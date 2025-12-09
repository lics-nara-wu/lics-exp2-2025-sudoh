###
### mylib_wordseg.py
### 単語分割のためのライブラリ
###

import sys
import regex as re

##### 課題
def extract_features (char_list, i):
    """ 文字のリストと分割候補点のインデックスから、該当する箇所の特徴量を抽出する
        入力例：
        char_list = ['私','は','学','生','で','す']
        i = 0
        出力例:
        X_list = ['L0:私', 'R0:は']

        「私」と「は」の間が分割されるかどうかを判断する特徴量として
        - 分割候補点のすぐ左にある文字が「私」→ "L0:私"
        - 分割候補点のすぐ右にある文字が「は」→ "R0:は"
        を考え、この分割候補点に対応する特徴量文字列として "L0:私 R0:は" を抽出した場合
    """
    X_list = [] ### 特徴量のリスト（後で空白区切りの文字列に整形する）
    
    ##### 特徴量抽出部 はじまり
    X_list.append( "L0:" + char_list[i] )
    X_list.append( "R0:" + char_list[i+1] )
    ##### 特徴量抽出部 おわり

    ##### ↑他にどういう特徴量が取り出せるか考えてみてください
    ##### このファイルの末尾に「文字の種類」を教えてくれる chartype という関数があります。
    ##### chartype('私') => '漢字'
    ##### chartype('は') => 'ひらがな'
    ##### chartype('ワ') => 'カタカナ'
    ##### chartype('1') => '数字'
    ##### chartype('A') => '欧字'
    ##### chartype('（') => 'その他'

    return X_list


##### 課題
def apply_wordseg (line, label_list):
    """ 文字列と分割ラベルの情報を使って分かち書きされた文字列を返す
    入力例
    line = "私はNara WUの学生です"
    label_list = ['|', '|', '-', '-', '-', '-', '|', '|', '-', '|', '-']
    望まれる出力
    line_wordseg = "私 は Nara WU の 学生 です"

    Tips:
    空白を追加するときには
    line_wordseg += " "
    とすれば良い
    """
    line_wordseg = ""

    for c in line:
        ##### このforの制御構造の中身を作成する(編集時次の行のpassは削除すること)
        pass

    return line_wordseg


def extract_features_from_labeled (line):
    """ ラベルつきの文字列から、単語分割のための特徴量文字列のリストとラベルのリストを返す（学習時用）
    """
    X_str_list = [] ### 最終的に取得する特徴量文字列のリスト
    y_list = [] ### 最終的に取得するラベルのリスト

    char_list  = [] ### 文字のリスト
    label_list = [] ### ラベルのリスト（ラベルは "-", "|", " "の三種類）

    labelflag = False ### 今ラベルを読む状態にあるかどうかのフラグ
    for c in line: ### 1文字ずつ走査する
        ### 文字とラベルは交互に出現するので、labelflagを使って区別する
        if labelflag:
            if c not in ["-", "|", " "]: ### ラベルのあるはずの位置に別の記号が来てしまっていた場合はデータエラーなのでこの後をスキップする
                #print (f"WARNING: skipping invalid line including a wrong label \"{c}\":\n{line}", file=sys.stderr)
                return X_str_list, y_list
            label_list.append(c)
            labelflag = False
        else:
            char_list.append(c)
            labelflag = True

    for i in range(len(label_list)): ### ラベルのリストのインデックス i ごとに特徴量とラベルの組を抽出
        if label_list[i] == " ":
            ### ラベルが空白文字のときは分類が必要ないのでスキップしてよい
            pass
        else:
            ### ラベルが "-" または "|" であれば特徴量文字列とラベルの組を抽出する
            X_list = extract_features(char_list, i)
            X_str_list.append(" ".join(X_list)) ### 特徴量を空白区切りの文字列に変換してリストに追加
            y_list.append(label_list[i]) ### ラベルをリストに追加

    return X_str_list, y_list


def extract_features_from_unlabeled (line):
    """ ラベルなしの（分かち書きされていない）文字列から特徴量文字列のリストを返す（実行時用）
    """
    X_str_list = []

    char_list = []
    for c in line:
        if c != " ": ### cが空白の場合は除外していることに注意
            char_list.append(c)

    for i in range(len(char_list)-1): ### 文字数は必要なラベル数より1多いことに注意
        X_list = extract_features(char_list, i)
        X_str_list.append(" ".join(X_list))

    return X_str_list


def chartype (char):
    if re.findall(r'\p{Han}+', char):
        return "漢字"
    elif re.findall(r'\p{Hiragana}+', char):
        return "ひらがな"
    elif re.findall(r'\p{Katakana}+', char):
        return "カタカナ"
    elif char.isalpha():
        return "欧字"
    elif char.isdigit():
        return "数字"
    else:
        return "その他"
