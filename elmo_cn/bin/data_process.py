# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : {NAME}.py
# Time    : 2019/5/13 0013 下午 3:27
"""



from glob import glob
import jieba
import operator


path = '../corpus/data*.txt'


def char_count_corpus(infile, outfile):
    '对每个字符进行统计'
    files = glob(infile)
    vocab = {}
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                for i in range(len(line)):
                    if line[i] in vocab:
                        vocab[line[i]] += 1
                    else:
                        vocab[line[i]] = 1
    sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1))
    with open(outfile, 'w', encoding='utf-8') as f:
        for key, val in sorted_vocab:
            f.write('{}\t{}\n'.format(key, val))


def test_char_count_corpus():
    inpattern = './corpus/data*.txt'
    outfile = './vocab/char_count_corps.txt'
    char_count_corpus(inpattern, outfile)

#test_char_count_corpus()


def gen_seg_files(infile):
    files = glob(infile)
    vocab_words_file = './vocab/word_count_corpus.txt'
    vocab_words = {}
    for file in files:
        outfile_chars = file.replace('.txt', '_seg_chars.txt')
        outfile_words = file.replace('.txt', '_seg_words.txt')
        list_lines_chars = []
        list_lines_words = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                'char 分词'
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                line_chars = ''
                for i in range(len(line)):
                    if line[i] == ' ':
                        continue
                    line_chars += line[i]
                    line_chars += ' '

                'word 分词'
                ss = jieba.cut(line, False, True)
                ss_new = []
                for s in ss:
                    if len(s) > 9:
                        print(s)
                        continue
                    ss_new.append(s)
                    if s in vocab_words:
                        vocab_words[s] += 1
                    else:
                        vocab_words[s] = 1
                line_words = ' '.join(ss_new)
                list_lines_chars.append(line_chars)
                list_lines_words.append(line_words)
        with open(outfile_chars, 'w', encoding='utf-8') as f:
            for s in list_lines_chars:
                f.write(s)
                f.write('\n')
        with open(outfile_words, 'w', encoding='utf-8') as f:
            for s in list_lines_words:
                f.write(s)
                f.write('\n')
    sorted_vocab_words = sorted(vocab_words.items(), key=operator.itemgetter(1))
    with open(vocab_words_file, 'w', encoding='utf-8') as f:
        for key, val in sorted_vocab_words:
            f.write(key)
            f.write('\t')
            f.write('{}'.format(val))
            f.write('\n')

def test_gen_seg_files():
    inpattern = './corpus/data*.txt'
    gen_seg_files(inpattern)

#test_gen_seg_files()


def gen_vocab_for_elmo():
    '生成char 词典和word 词典'
    vocab_chars_infile = './vocab/char_count_corpus.txt'
    vocab_words_infile = './vocab/word_count_corpus.txt'
    vocab_chars_outfile = './vocab/vocab_seg_chars_elmo.txt'
    vocab_words_outfile = './vocab/vocab_seg_words_elmo.txt'
    set_chars = set(['<S>', '</S>', '<UNK>'])
    set_words = set(['<S>', '</S>', '<UNK>'])
    with open(vocab_chars_infile, 'r', encoding='utf-8') as fin:
        with open(vocab_chars_outfile, 'w', encoding='utf-8') as fout:
            for line in fin.readlines():
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                ss = line.split()
                if len(ss) != 2:
                    continue
                set_chars.add(ss[0])
                fout.write(ss[0])
                fout.write('\n')
    with open(vocab_words_infile, 'r', encoding='utf-8') as fin:
        with open(vocab_words_outfile, 'w', encoding='utf-8') as fout:
            for line in fin.readlines():
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                ss = line.split()
                if len(ss) != 2:
                    continue
                set_words.add(ss[0])
                fout.write(ss[0])
                fout.write('\n')
    with open(vocab_chars_outfile, 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(set_chars))
    with open(vocab_words_outfile, 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(set_words))


#gen_vocab_for_elmo()


#统计所有分词中包含的字的最大个数：
def stat_max_length_in_words(infile):
    with open(infile, 'r', encoding='utf-8') as f:
        max_length = 0
        list = []
        for line in f.readlines():
            line = line.strip()
            if line is None or len(line) == 0:
                continue
            if max_length < len(line):
                max_length = len(line)
            if len(line) > 5:
                list.append(line)
    print('max length in words is: {}'.format(max_length))
    outfile = './words_freq_stat.txt'
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write('\n'.join(list))

def test_stat_max_length_in_words():
    infile = '../data/vocab_seg_words_elmo.txt'
    stat_max_length_in_words(infile)

# 统计token总数目
def stat_tokens_num(inpattern):
    files = glob(inpattern)
    total_cnt = 0
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                ss = line.split()
                total_cnt += len(ss)
    print('total tokens number is: {}'.format(total_cnt))

def test_stat_tokens_num():
    # total tokens number is: 4775300
    inpattern = '../data/*_seg_words.txt'
    stat_tokens_num(inpattern)


#input_prefix = ['./corpus/data1.txt', './corpus/data2.txt']
input_prefix = ['./corpus/data1_seg_chars.txt', './corpus/data2_seg_chars.txt']
#input_prefix = ['./corpus/data1_seg_words.txt', './corpus/data2_seg_words.txt']
#outfile = './corpus/example.txt'
outfile = './corpus/example_seg_chars.txt'
#outfile = './corpus/example_seg_words.txt'

def merge_files_to_one_file(input_prefix, outfile):
    #file_list = glob(input_prefix)
    file_list = input_prefix
    contents = []
    print(file_list)
    for file in file_list:
        with open(file, 'r', encoding='utf8') as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or line == '':
                    continue
                contents.append(line)

    with open(outfile, 'w', encoding='utf8') as f:
        f.write('\n'.join(contents))


merge_files_to_one_file(input_prefix, outfile)












