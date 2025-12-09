#!/usr/bin/env python3

import sys
import re
from sklearn.metrics import accuracy_score
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-r", "--reference", type=str, metavar="REFERENCE", help="reference tokenization with labels")
parser.add_argument("hypothesis", type=str, metavar="HYPOTHESIS", help="hypothesis tokenization WITHOUT labels")
args = parser.parse_args()


def extract_labels_from_labeled (line):
    label_list = []
    labelflag = False
    for c in line: ### 1文字ずつ走査する
        ### 文字とラベルは交互に出現するので、labelflagを使って区別する
        if labelflag:
            if c not in ["-", "|", " "]: ### ラベルのあるはずの位置に別の記号が来てしまっていた場合はデータエラーなので残りをスキップする
                #print (f"WARNING: skipping invalid line including a wrong label \"{c}\":\n{line}\n{char_list}\n{label_list}", file=sys.stderr)
                return None
            label_list.append(c)
            labelflag = False
        else:
            labelflag = True
    return label_list


def extract_labels_from_unlabeled (line):
    label_list = []
    for i in range(len(line)):
        if line[i] == " ":
            label_list.append("|")
        elif i > 0 and line[i-1] != " ":
            label_list.append("-")
    return label_list


with open(args.reference, 'rt') as rh1, open(args.hypothesis, 'rt') as rh2:
    y_test = []
    y_predict = []
    for rline, hline in zip(rh1, rh2):
        rline = rline.rstrip("\n")
        hline = re.sub(r' +', ' ', hline.rstrip("\n").strip(" "))
        rlabels = extract_labels_from_labeled(rline)
        hlabels = extract_labels_from_unlabeled(hline)
        if len(rlabels) != len(hlabels):
            print (rline)
            print (rlabels)
            print (hline)
            print (hlabels)
            raise RuntimeError
        else:
            if " " in rlabels:
                for i in range(len(rlabels)-1, -1, -1):
                    if rlabels[i] == " ":
                        rlabels.pop(i)
                        hlabels.pop(i)
            y_test    += rlabels
            y_predict += hlabels

    test_accuracy = accuracy_score(y_test, y_predict)
    print (f"Test accuracy: {100*test_accuracy:.3g} %")
