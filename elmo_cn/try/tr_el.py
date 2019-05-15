# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : tr_el.py
# Time    : 2019/5/15 0015 下午 1:47
"""


from bilm.training import load_vocab, train_with_signal_core
from bilm.data import BidirectionalLMDataset


tf_save_dir = './out/'
tf_log_dir = './out/'
vocab_file = '../vocab/vocab_seg_words_elmo.txt'
inpattern = '../corpus/example_seg_words.txt'
vocab = load_vocab(vocab_file, 10)
batch_size = 64
n_gpus = 1
n_train_tokens = 29723
options = {
    'bidirectional': True,

    'char_cnn': {'activation': 'relu',
                 'embedding': {'dim': 16},
                 'filters': [[1, 32],
                             [2, 32],
                             [3, 64],
                             [4, 128],
                             [5, 256],
                             [6, 512],
                             [7, 1024]],
                 'max_characters_per_token': 10,
                 'n_characters': 105047,
                 'n_highway': 2},

    'dropout': 0.1,

    'lstm': {
        'cell_clip': 3,
        'dim': 4096,
        'n_layers': 2,
        'proj_clip': 3,
        'projection_dim': 512,
        'use_skip_connections': True},

    'all_clip_norm_val': 10.0,

    'n_epochs': 10,
    'n_train_tokens': n_train_tokens,
    'batch_size': batch_size,
    'n_tokens_vocab': vocab.size,
    'unroll_steps': 20,
    'n_negative_samples_batch': 8192,
}



data = BidirectionalLMDataset(inpattern, vocab, test=False, shuffle_on_load=True)

train_with_signal_core(options, data, tf_save_dir, tf_log_dir, restart_ckpt_file=None)








