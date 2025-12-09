import os
import json
from csv import DictReader
from argparse import ArgumentParser

# wrime ver2 データの置き場所（須藤のディレクトリ内）
WRIME_V2 = "/export/home/ics/sudoh/Project/Exp2/2024/wrime/wrime-ver2.tsv"

parser = ArgumentParser()
parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing output file.")
parser.add_argument("-t", "--tokenize", action="store_true", help="Tokenize sentences using MeCab (mecab-python3==1.0.4 and unidic-lite are required).")
parser.add_argument("output", type=str, metavar="FILE", help="Output file.")
args = parser.parse_args()

tagger = None
if args.tokenize:
    from MeCab import Tagger
    tagger = Tagger("-Owakati")

if os.path.exists(args.output) and not args.force:
    raise RuntimeError (f"ERROR: {args.output} already exists.")
else:
    DATA = {'train':[], 'dev':[], 'test':[]}
    # wrime ver2 データを読み込む
    with open(WRIME_V2, 'rt') as rfp:
        tsvreader = DictReader(rfp, delimiter='\t')
        for row in tsvreader:
            sentence = row['Sentence']
            if args.tokenize:
                sentence = tagger.parse(sentence).rstrip('\n')
            DATA[row['Train/Dev/Test']].append({'Sentence':sentence, 'Writer_Sentiment':int(row['Writer_Sentiment'])})
    
    with open(args.output, 'wt') as wfp:
        json.dump(DATA, wfp, ensure_ascii=False)
