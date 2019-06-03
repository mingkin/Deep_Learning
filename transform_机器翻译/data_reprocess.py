# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : data_process.py
# Time    : 2019/5/31 0031 下午 5:29
© 2019 Ming. All rights reserved. Powered by King
"""


import os
import errno
import sentencepiece as spm
import re
from hparams import Hparams
import logging
import os
import regex
from collections import Counter


logging.basicConfig(level=logging.INFO)


def prepro(hp):
    """Load raw data -> Preprocessing -> Segmenting with sentencepice
    hp: hyperparams. argparse.
    """
    logging.info("# Check if raw files exist")
    train1 = "../de-en/train.tags.de-en.de"
    train2 = "../de-en/train.tags.de-en.en"
    eval1 = "../de-en/IWSLT16.TED.tst2013.de-en.de.xml"
    eval2 = "../de-en/IWSLT16.TED.tst2013.de-en.en.xml"
    test1 = "../de-en/IWSLT16.TED.tst2014.de-en.de.xml"
    test2 = "../de-en/IWSLT16.TED.tst2014.de-en.en.xml"
    for f in (train1, train2, eval1, eval2, test1, test2):
        if not os.path.isfile(f):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), f)

    logging.info("# Preprocessing")
    prepro_train1, prepro_train2 = [], []
    with open(train1, 'r', encoding='utf8') as f1:
        for line in f1.readlines():
            if not line.startswith("<"):
                prepro_train1.append(line.split("\n")[0].strip())

    with open(train2, 'r', encoding='utf8') as f2:
        for line in f2.readlines():
            if not line.startswith("<"):
                prepro_train2.append(line.split("\n")[0].strip())
    # train
    assert len(prepro_train1)==len(prepro_train2)# "Check if train source and target files match."

    # eval
    prepro_eval1, prepro_eval2 = [], []

    with open(eval1, 'r', encoding='utf8') as f3:
        for line in f3.readlines():
            if line.startswith("<seg id"):
                prepro_eval1.append(re.sub("<[^>]+>", "", line.split("\n")[0]).strip())

    with open(eval2, 'r', encoding='utf8') as f4:
        for line in f4.readlines():
            if  line.startswith("<seg id"):
                prepro_eval2.append(re.sub("<[^>]+>", "", line.split("\n")[0]).strip())


    assert len(prepro_eval1) == len(prepro_eval2)# "Check if eval source and target files match."
    # test
    prepro_test1, prepro_test2 = [], []

    with open(eval1, 'r', encoding='utf8') as f5:
        for line in f5.readlines():
            if line.startswith("<seg id"):
                prepro_test1.append(re.sub("<[^>]+>", "", line.split("\n")[0]).strip())

    with open(eval2, 'r', encoding='utf8') as f6:
        for line in f6.readlines():
            if line.startswith("<seg id"):
                prepro_test2.append(re.sub("<[^>]+>", "", line.split("\n")[0]).strip())


    assert len(prepro_test1) == len(prepro_test2) #"Check if test source and target files match."

    logging.info("Let's see how preprocessed data look like")
    logging.info("# write preprocessed files to disk")
    os.makedirs("iwslt2016/prepro", exist_ok=True)
    def _write(sents, fname):
        with open(fname, 'w', encoding='utf8') as fout:
            fout.write("\n".join(sents))

    _write(prepro_train1, "iwslt2016/prepro/train.de")
    _write(prepro_train2, "iwslt2016/prepro/train.en")
    _write(prepro_train1+prepro_train2, "iwslt2016/prepro/train")
    _write(prepro_eval1, "iwslt2016/prepro/eval.de")
    _write(prepro_eval2, "iwslt2016/prepro/eval.en")
    _write(prepro_test1, "iwslt2016/prepro/test.de")
    _write(prepro_test2, "iwslt2016/prepro/test.en")

    logging.info("# Train a joint BPE model with sentencepiece")
    os.makedirs("iwslt2016/segmented", exist_ok=True)
    train = '--input=iwslt2016/prepro/train --pad_id=0 --unk_id=1 \
             --bos_id=2 --eos_id=3\
             --model_prefix=iwslt2016/segmented/bpe --vocab_size={} \
             --model_type=bpe'.format(hp.vocab_size)
    spm.SentencePieceTrainer.Train(train)

    logging.info("# Load trained bpe model")
    sp = spm.SentencePieceProcessor()
    sp.Load("iwslt2016/segmented/bpe.model")

    logging.info("# Segment")
    def _segment_and_write(sents, fname):
        with open(fname, "w", encoding='utf8') as fout:
            for sent in sents:
                pieces = sp.EncodeAsPieces(sent)
                fout.write(" ".join(pieces) + "\n")

    _segment_and_write(prepro_train1, "iwslt2016/segmented/train.de.bpe")
    _segment_and_write(prepro_train2, "iwslt2016/segmented/train.en.bpe")
    _segment_and_write(prepro_eval1, "iwslt2016/segmented/eval.de.bpe")
    _segment_and_write(prepro_eval2, "iwslt2016/segmented/eval.en.bpe")
    _segment_and_write(prepro_test1, "iwslt2016/segmented/test.de.bpe")

    logging.info("Let's see how segmented data look like")
    print("train1:", open("iwslt2016/segmented/train.de.bpe", 'r', encoding='utf8').readline())
    print("train2:", open("iwslt2016/segmented/train.en.bpe", 'r', encoding='utf8').readline())
    print("eval1:", open("iwslt2016/segmented/eval.de.bpe", 'r', encoding='utf8').readline())
    print("eval2:", open("iwslt2016/segmented/eval.en.bpe", 'r', encoding='utf8').readline())
    print("test1:", open("iwslt2016/segmented/test.de.bpe", 'r', encoding='utf8').readline())


def make_vocab(fpath, fname):
    '''Constructs vocabulary.

    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.

    Writes vocabulary line by line to `preprocessed/fname`
    '''
    text = open(fpath, 'r', 'utf-8').read()
    text = regex.sub("[^\s\p{Latin}']", "", text)
    words = text.split()
    word2cnt = Counter(words)
    if not os.path.exists('preprocessed'): os.mkdir('preprocessed')
    with open('preprocessed/{}'.format(fname), 'w', 'utf-8') as fout:
        fout.write(
            "{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))


if __name__ == '__main__':
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    prepro(hp)
    # make_vocab(hp.source_train, "de.vocab.tsv")
    # make_vocab(hp.target_train, "en.vocab.tsv")
    logging.info("Done")






