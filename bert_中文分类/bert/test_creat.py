# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : {NAME}.py
# Time    : 2019/3/15 0015 下午 3:11
"""

import collections
import tokenization
import tensorflow as tf
import random
rng = random.Random(1)


" 创建实例"
input_file = 'sample_text.txt'
input_files = []
for input_pattern in input_file.split(","):
    print(input_pattern)
    input_files.extend(tf.gfile.Glob(input_pattern))

print(input_files)
tokenizer = tokenization.FullTokenizer(
      vocab_file='../chinese_L-12_H-768_A-12/vocab.txt', do_lower_case=True)


all_documents = [[]]


for input_file in input_files:
    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
        line = tokenization.convert_to_unicode(reader.readline())
        #print(line)
        if not line:
          print('======')
          break
        line = line.strip()
        # Empty lines are used as document delimiters
        if not line:
          all_documents.append([])
        tokens = tokenizer.tokenize(line)

        if tokens:
          all_documents[-1].append(tokens)


all_documents = [x for x in all_documents if x]
rng.shuffle(all_documents)
#print(all_documents)
vocab_words = list(tokenizer.vocab.keys())
instances = []
print(vocab_words)

max_seq_length=128
max_predictions_per_seq=20
masked_lm_prob=0.15
short_seq_prob=0.1

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()

def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
  """Creates the predictions for the masked LM objective.
  这部分对token进行随机mask。这部分是BERT的创新点之二，随机遮蔽。为了防止双向模型在多层之后“看到自己”。这里对一部分词进行随机遮蔽，并在预训练中进行预测。遮蔽方案：
  1.以80%的概率直接变成[MASK]
  2.以10%的概率保留原词
  3.以10%的概率在词典中随机找一个词替代
  返回值：经过随机遮蔽后的（词，遮蔽位置，遮蔽前原词）
  """

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    cand_indexes.append(i)

  rng.shuffle(cand_indexes)#打乱顺序

  output_tokens = list(tokens)

  masked_lm = collections.namedtuple("masked_lm", ["index", "label"])  # p定义一个名为masked_lm的元组，里面有两个属性

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))#所有要mask的词的数量为定值，取两个定义好参数的最小值

  masked_lms = []
  covered_indexes = set()
  for index in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    if index in covered_indexes:
      continue
    covered_indexes.add(index)#要被mask的词的index

    masked_token = None
    # 80% of the time, replace with [MASK]
    if rng.random() < 0.8:
      masked_token = "[MASK]"
    else:
      # 10% of the time, keep original
      if rng.random() < 0.5:
        masked_token = tokens[index]
      # 10% of the time, replace with random word
      else:
        masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

    output_tokens[index] = masked_token #用masked_token替换原词

    masked_lms.append(masked_lm(index=index, label=tokens[index]))

  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)#被mask的index
    masked_lm_labels.append(p.label)#被mask的label（即原词）


  return (output_tokens, masked_lm_positions, masked_lm_labels)

class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
               is_random_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "is_random_next: %s\n" % self.is_random_next
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def create_instances_from_document(
    all_documents, document_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
  """Creates `TrainingInstance`s for a single document.
   这部分是生成训练数据的具体过程，对每条数据生成TrainingInstance。这里的每条数据其实包含两个句子的信息。
   TrainingInstance包括tokens,segement_ids,is_random_next,masked_lm_positions,masked_lm_labels。
   下面给出这些属性的含义
    tokens：词
    segement_id：句子编码 第一句为0 第二句为1
    is_random_next：第二句是随机查找，还是为第一句的下文
    masked_lm_positions：tokens中被mask的位置
    masked_lm_labels：tokens中被mask的原来的词
    本部分含有BERT的创新点之一：下一句预测 类标的生成
    返回值：instances
  """
  document = all_documents[document_index]
  #第i个句子
  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.
  target_seq_length = max_num_tokens

  if rng.random() < short_seq_prob: #产生一个随机数如果小于short_seq_prob 则产生一个较短的训练序列
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  instances = []
  current_chunk = [] #产生训练集的候选集
  current_length = 0
  i = 0
  print(len(document))
  while i < len(document):
    print(i)
    segment = document[i]
    current_chunk.append(segment)
    current_length += len(segment)

    if i == len(document) - 1 or current_length >= target_seq_length:
      if current_chunk:
        # `a_end` is how many segments from `current_chunk` go into the `A`
        # (first) sentence.
        a_end = 1
        if len(current_chunk) >= 2:
          a_end = rng.randint(1, len(current_chunk) - 1)#从current_chunk中随机选出一个文档作为句子1的截止文档

        tokens_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])#将截止文档之前的文档都加入到tokens_a

        tokens_b = []
        # Random next
        is_random_next = False
        if len(current_chunk) == 1 or rng.random() < 0.5:
          #候选集只有一句的情况则随机抽取句子作为句子2；或以0.5的概率随机抽取句子作为句子2

          is_random_next = True
          target_b_length = target_seq_length - len(tokens_a)

          # This should rarely go for more than one iteration for large
          # corpora. However, just to be careful, we try to make sure that
          # the random document is not the same as the document
          # we're processing.
          for _ in range(10):
            random_document_index = rng.randint(0, len(all_documents) - 1)
            if random_document_index != document_index:
              break

          random_document = all_documents[random_document_index]#随机找一个文档作为截止文档
          random_start = rng.randint(0, len(random_document) - 1)#随机找一个初始文档
          for j in range(random_start, len(random_document)):
            tokens_b.extend(random_document[j])#将随机文档加入到token_b
            if len(tokens_b) >= target_b_length:
              break
          # We didn't actually use these segments so we "put them back" so
          # they don't go to waste.
          num_unused_segments = len(current_chunk) - a_end

          i -= num_unused_segments
        # Actual next
        else:
          is_random_next = False #以第1句的后续作为句子2
          for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)#对两个句子进行长度剪裁

        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
          tokens.append(token)
          segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
          tokens.append(token)
          segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        print(tokens)
        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
             tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)#对token创建mask
        print(tokens)
        print(masked_lm_positions)
        print(masked_lm_labels)

        instance = TrainingInstance(
            tokens=tokens,
            segment_ids=segment_ids,
            is_random_next=is_random_next,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)

      current_chunk = []
      current_length = 0
    i += 1

  return instances



for _ in range(2):
    print(all_documents)
    for document_index in range(len(all_documents)):
      instances.extend(
          create_instances_from_document(
              all_documents, document_index, max_seq_length, short_seq_prob,
              masked_lm_prob, max_predictions_per_seq, vocab_words, rng))
      print(instances)
    rng.shuffle(instances)



" modeling 模块"



























