# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : bilm_data_try.py
# Time    : 2019/5/14 0014 下午 2:31
"""



from bilm.data import Vocabulary, UnicodeCharsVocabulary, LMDataset, BidirectionalLMDataset



def test_vocabulary():
    print('chars encoder senctence:')
    vocab_file = '../vocab/vocab_seg_chars_elmo.txt'
    vocab_chars = Vocabulary(vocab_file, True)

    print('====> bos: {}'.format(vocab_chars.bos))
    print('====> eos: {}'.format(vocab_chars.eos))
    print('====> unk: {}'.format(vocab_chars.unk))
    print('====> size: {}'.format(vocab_chars.size))

    word = '刀'
    print(word+'====> word to id: {}'.format(vocab_chars.word_to_id(word)))
    word = '阿'
    print(word+'====> word to id: {}'.format(vocab_chars.word_to_id(word)))

    id = 234
    print(str(id)+'====> id to word: {}'.format(vocab_chars.id_to_word(id)))
    id = 2342
    print(str(id)+'====> id to word: {}'.format(vocab_chars.id_to_word(id)))

    word = '云逸听到这声音十分耳熟'
    print(word+'====> encoded result without split: {}'.format(vocab_chars.encode(word, split=False)))
    print(word+'====> encoded result with split: {}'.format(vocab_chars.encode(word, split=True)))
    word = ' 云 逸 听 到 这 声 音 十 分 耳 熟'
    print(word+'====> encoded result without split: {}'.format(vocab_chars.encode(word)))
    print(word+'====> encoded result with split: {}'.format(vocab_chars.encode(word, split=True)))
    print(word+'====> encoded result with split with reverse: {}'.format(vocab_chars.encode(word, split=True, reverse=True)))


    # input words
    print('words encoder senctence: ')
    vocab_file = '../vocab/vocab_seg_words_elmo.txt'
    vocab_chars = Vocabulary(vocab_file, True)
    print('====> bos: {}'.format(vocab_chars.bos))
    print('====> eos: {}'.format(vocab_chars.eos))
    print('====> unk: {}'.format(vocab_chars.unk))
    print('====> size: {}'.format(vocab_chars.size))
    word = '云逸听到这声音十分耳熟'
    print('====> word to id: {}'.format(vocab_chars.word_to_id(word)))
    word = ' 云逸 听到 这 声音 十分 耳熟'
    print('====> word to id: {}'.format(vocab_chars.word_to_id(word)))

    id = 234
    print('====> id to word: {}'.format(vocab_chars.id_to_word(id)))
    id = 2344
    print('====> id to word: {}'.format(vocab_chars.id_to_word(id)))

    word = '云逸听到这声音十分耳熟'
    print('====> encoded result without split: {}'.format(vocab_chars.encode(word, split=False)))
    print('====> encoded result with split: {}'.format(vocab_chars.encode(word, split=True)))
    word = '云逸 听到 这 声音 十分 耳熟'
    print('====> encoded result without split: {}'.format(vocab_chars.encode(word, split=False)))
    print('====> encoded result with split: {}'.format(vocab_chars.encode(word, split=True)))
    print('====> encoded result with split with reverse: {}'.format(vocab_chars.encode(word, split=True, reverse=True)))


#test_vocabulary()


def test_unicode():
    '''
    UnicodeCharsVocabulary
    '''
    print('UnicodeCharsVocabulary:')
    vocab_file = '../vocab/vocab_seg_words_elmo.txt'
    vocab_file1 = '../vocab/vocab_seg_chars_elmo.txt'
    vocab_chars = Vocabulary(vocab_file, True)
    vocab_unicodechars = UnicodeCharsVocabulary(vocab_file,  max_word_length=10, validate_file=True)
    print('====> bos: {}'.format(vocab_chars.bos))
    print('====> eos: {}'.format(vocab_chars.eos))
    print('====> unk: {}'.format(vocab_chars.unk))
    print('====> size: {}'.format(vocab_chars.size))

    word = '阿道夫'
    print('====> word to id: {}'.format(vocab_chars.word_to_id(word)))
    word = '阿'
    print('====> word to id: {}'.format(vocab_chars.word_to_id(word)))

    id = 234
    print('====> id to word: {}'.format(vocab_chars.id_to_word(id)))
    id = 234
    print('====> id to word: {}'.format(vocab_chars.id_to_word(id)))

    print('====> vocab_unicodechars size: {}'.format(vocab_unicodechars.size))


    print('====>vocab_unicodechars bos: {}'.format(vocab_unicodechars.bos))
    print('====>vocab_unicodechars eos: {}'.format(vocab_unicodechars.eos))
    print('====>vocab_unicodechars unk: {}'.format(vocab_unicodechars.unk))
    print('====>vocab_unicodechars id to chars: {}'.format(vocab_unicodechars._id_to_word))
    print('====>vocab_unicodechars word chars ids: {}'.format(vocab_unicodechars.word_char_ids))
    #print('====> max word length: {}'.format(vocab_unicodechars.word_char_ids(0)))
    words = '云逸 听到 这 声音 十分 耳熟 ，'
    print('====> word \t{}\t encoded result: {}'.format(words, vocab_chars.encode(words)))
    print('====> word \t{}\t to char ids: {}'.format(words, vocab_unicodechars.word_to_char_ids(words)))
    print('====> word \t{}\t encoded chars id result: {}'.format(words, vocab_unicodechars.encode_chars(words)))

    ids = [1234, 3234, 22, 34, 3413, 21, 345]
    print('====> decode \t{}\t to words: {}'.format(ids, vocab_unicodechars.decode(ids)))


#test_unicode()


def test_lmdataset():
    '''
    test LMDataset
    '''
    print('LMDataset:')
    vocab_file = '../vocab/vocab_seg_words_elmo.txt'
    vocab_unicodechars = UnicodeCharsVocabulary(vocab_file, max_word_length=10, validate_file=True)
    filepattern = '../corpus/example_seg_words.txt'
    lmds = LMDataset(filepattern, vocab_unicodechars, test=True)
    batch_size = 128
    n_gpus = 1
    unroll_steps = 10
    data_gen = lmds.iter_batches(batch_size * n_gpus, unroll_steps)
    jump_cnt = 0
    for num, batch in enumerate(data_gen, start=1):
        jump_cnt += 1
        if jump_cnt > 10:
            break
        print('====> iter [{}]\ttoken ids shape: {}'.format(num, batch['token_ids'].shape))
        print('====> iter [{}]\ttokens characters shape: {}'.format(num, batch['tokens_characters'].shape))
        print('====> iter [{}]\tnext token ids shape: {}'.format(num, batch['next_token_id'].shape))

test_lmdataset()

def test_bi():
    '''
    UE for BidirectionalLMDataset
    '''
    print('BidirectionalLMDataset:')
    vocab_file = '../vocab/vocab_seg_words_elmo.txt'
    vocab_unicodechars = UnicodeCharsVocabulary(vocab_file, max_word_length=10, validate_file=True)
    filepattern = '../corpus/example_seg_words.txt'
    bilmds = BidirectionalLMDataset(filepattern, vocab_unicodechars, test=True)
    batch_size = 128
    n_gpus = 1
    unroll_steps = 10
    data_gen = bilmds.iter_batches(batch_size * n_gpus, unroll_steps)
    jump_cnt = 0
    for num, batch in enumerate(data_gen, start=1):
        jump_cnt += 1
        if jump_cnt > 10:
            break
        print('\n')
        print('====> iter [{}]\ttoken ids shape: {}'.format(num, batch['token_ids'].shape))
        print('====> iter [{}]\ttokens characters shape: {}'.format(num, batch['tokens_characters'].shape))
        print('====> iter [{}]\tnext token ids shape: {}'.format(num, batch['next_token_id'].shape))
        print('====> iter [{}]\ttoken ids reverse shape: {}'.format(num, batch['token_ids_reverse'].shape))
        print('====> iter [{}]\ttokens characters reverse shape: {}'.format(num, batch['tokens_characters_reverse'].shape))
        print('====> iter [{}]\tnext token ids reverse shape: {}'.format(num, batch['next_token_id_reverse'].shape))

test_bi()