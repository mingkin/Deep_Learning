# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : {NAME}.py
# Time    : 2019/5/15 0015 上午 11:13
"""


import numpy as np
max_word_length =10

batch_size, num_steps = 20, 35
cur_stream = [None] * batch_size
no_more_data = False
generator = [
[0,   803,  4304,  8102, 28154, 16625, 27275, 29019,  5633,
22650, 17985, 13946, 22650, 29201, 26219, 15609,  9991, 22650,
22426,   617, 11328, 22650, 27548, 20076, 14090, 11721,  1419,
4304,  8609, 25113,  2279, 28792, 21476,     1],
[[258, 256, 259, 260, 260, 260, 260, 260, 260, 260],
[258, 228, 186, 145, 233, 128, 184, 259, 260, 260],
[258, 231, 154, 132, 259, 260, 260, 260, 260, 260],
[258, 229, 133, 131, 259, 260, 260, 260, 260, 260],
[258, 231, 165, 158, 259, 260, 260, 260, 260, 260],
[258, 229, 143, 136, 259, 260, 260, 260, 260, 260],
[258, 230, 152, 175, 259, 260, 260, 260, 260, 260],
[258, 228, 189, 149, 229, 133, 182, 259, 260, 260],
[258, 229, 188, 186, 229, 164, 167, 259, 260, 260],
[258, 239, 188, 140, 259, 260, 260, 260, 260, 260],
[258, 231, 165, 158, 229, 191, 181, 259, 260, 260],
[258, 228, 184, 128, 229, 138, 168, 259, 260, 260],
[258, 239, 188, 140, 259, 260, 260, 260, 260, 260],
[258, 231, 171, 139, 229, 141, 179, 259, 260, 260],
[258, 230, 145, 134, 232, 132, 177, 259, 260, 260],
[258, 228, 186, 134, 259, 260, 260, 260, 260, 260],
[258, 228, 184, 141, 233, 128, 130, 259, 260, 260],
[258, 239, 188, 140, 259, 260, 260, 260, 260, 260],
[258, 230, 151, 139, 229, 141, 179, 259, 260, 260],
[258, 232, 132, 154, 228, 184, 139, 259, 260, 260],
[258, 230, 187, 145, 231, 167, 187, 259, 260, 260],
[258, 239, 188, 140, 259, 260, 260, 260, 260, 260],
[258, 232, 135, 170, 231, 132, 182, 232, 128, 259],
[258, 229, 156, 176, 259, 260, 260, 260, 260, 260],
[258, 230, 150, 189, 229, 177, 149, 259, 260, 260],
[258, 229, 135, 186, 233, 185, 164, 259, 260, 260],
[258, 229, 189, 162, 230, 139, 179, 259, 260, 260],
[258, 231, 154, 132, 259, 260, 260, 260, 260, 260],
[258, 233, 185, 164, 229, 135, 140, 230, 179, 259],
[258, 233, 151, 170, 231, 167, 187, 259, 260, 260],
[258, 229, 136, 176, 259, 260, 260, 260, 260, 260],
[258, 228, 184, 128, 230, 151, 129, 259, 260, 260],
[258, 227, 128, 130, 259, 260, 260, 260, 260, 260],
[258, 257, 259, 260, 260, 260, 260, 260, 260, 260]]]

while True:
    inputs = np.zeros([batch_size, num_steps], np.int32)
    if max_word_length is not None:
        char_inputs = np.zeros([batch_size, num_steps, max_word_length],
                            np.int32)
    else:
        char_inputs = None
    targets = np.zeros([batch_size, num_steps], np.int32)

    for i in range(batch_size):
        cur_pos = 0    # cur_stream当前位置 num_steps窗口大小
        print('batch_size', i)
        while cur_pos < num_steps:
            # cur_stream当前位置 < num_steps窗口大小  每个句子只取一个固定窗口
            if cur_stream[i] is None or len(cur_stream[i][0]) <= 1:
                try:
                    cur_stream[i] = list(generator) #把不定长句子加入cur_stream[i]
                except StopIteration:
                    # No more data, exhaust current streams and quit
                    no_more_data = True
                    break
                print('how', len(cur_stream[i][0]) - 1, num_steps - cur_pos)
                how_many = min(len(cur_stream[i][0]) - 1, num_steps - cur_pos)
                print('how_many', how_many)
                next_pos = cur_pos + how_many
                # 一次读一个窗口的数据 inputs [batch_size, num_steps]
                print('next_pos', next_pos)
                inputs[i, cur_pos:next_pos] = cur_stream[i][0][:how_many]
                print('inputs', inputs[i])
                if max_word_length is not None:
                    char_inputs[i, cur_pos:next_pos] = cur_stream[i][1][
                                                       :how_many]
                targets[i, cur_pos:next_pos] = cur_stream[i][0][1:how_many + 1]
                print('char_outputs', char_inputs[i])
                print('outputs',targets[i])
                cur_pos = next_pos
                print('cur_pos', cur_pos)
                cur_stream[i][0] = cur_stream[i][0][how_many:]
                if max_word_length is not None:
                    cur_stream[i][1] = cur_stream[i][1][how_many:]

            if no_more_data:
                # There is no more data.  Note: this will not return data
                # for the incomplete batch
                break

            X = {'token_ids': inputs, 'tokens_characters': char_inputs,
                 'next_token_id': targets}





