# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import os
import random
import re
import json

from hbconfig import Config
import numpy as np
# import tensorflow as tf
from tqdm import tqdm
from six.moves import cPickle
from collections import Counter

MODELDIR = "/home/dingo/lib_data/ltp_data_v3.4.0"
from pyltp import Segmentor

segmentor = Segmentor()
segmentor.load(os.path.join(MODELDIR, "cws.model"))

# segmentor = None


def get_question_answers():
    """ Divide the dataset into two sets: questions and answers. """
    file_path = os.path.join(Config.data.base_path, Config.data.line_fname)

    questions = []
    answers = []
    with open(file_path, 'r', encoding='utf-8') as f:
        session = []
        for line in f.readlines():
            if line.strip() == '':
                if session == []:
                    continue
                questions.append(session[: -1])
                answers.append(session[-1])
                session = []
            else:
                session.append(line.strip().lower())
    if len(session) != 0:
        questions.append(session[: -1])
        answers.append(session[-1])
    assert len(questions) == len(answers)
    return questions, answers


def prepare_dataset(questions, answers):
    # create path to store all the train & test encoder & decoder
    make_dir(Config.data.base_path + Config.data.processed_path)

    # random convos to create the test set
    test_ids = random.sample([i for i in range(len(questions))], Config.data.testset_size)

    filenames = ['train.enc', 'train.dec', 'test.enc', 'test.dec']
    files = []
    for filename in filenames:
        files.append(open(os.path.join(Config.data.base_path, Config.data.processed_path, filename), 'w', encoding='utf-8'))

    for i in tqdm(range(len(questions))):

        question = questions[i]
        answer = answers[i]

        if i in test_ids:
            files[2].write((json.dumps(question, ensure_ascii=False) + "\n"))
            files[3].write((answer + '\n'))
        else:
            files[0].write((json.dumps(question, ensure_ascii=False) + '\n'))
            files[1].write((answer + '\n'))

    for file in files:
        file.close()


def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def basic_tokenizer(line, normalize_digits=True):
    """ A basic tokenizer to tokenize text into tokens.
    Feel free to change this to suit your need. """
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    words = []
    _WORD_SPLIT = re.compile("([.,!?\"'-<>:;)(])")
    _DIGIT_RE = re.compile(r"\d")
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            if normalize_digits:
                token = re.sub(_DIGIT_RE, '#', token)
            words.append(token)
    return words


def build_vocab(in_fname, out_fname, normalize_digits=True):
    embed = {}
    print('Loading word embedding...')
    with open(os.path.join(Config.data.base_path, Config.data.word_emb), 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            items = line.split(' ')
            word = items[0]
            vec = items[1:]
            embed[word] = [float(x) for x in vec]

    print("Count each vocab frequency ...")
    vocab = {}
    oov = {}
    def count_vocab(fname, multi=False):
        with open(fname, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                if multi:
                    us = json.loads(line)
                    for u in us:
                        # for token in segmentor.segment(u):
                        for token in u.strip().split('\t'):
                            # token = token.strip()
                            if not token in embed:
                                if not token in oov:
                                    oov[token] = 0
                                oov[token] += 1
                                continue
                            if not token in vocab:
                                vocab[token] = 0
                            vocab[token] += 1
                else:
                    # for token in segmentor.segment(line):
                    for token in line.strip().split('\t'):
                        # token = token.strip()
                        if not token in embed:
                            if not token in oov:
                                oov[token] = 0
                            oov[token] += 1
                            continue
                        if not token in vocab:
                            vocab[token] = 0
                        vocab[token] += 1

    in_path = os.path.join(Config.data.base_path, Config.data.processed_path, in_fname)
    out_path = os.path.join(Config.data.base_path, Config.data.processed_path, out_fname)

    count_vocab(in_path, multi=True)
    count_vocab(out_path)

    print("total vocab size:", len(vocab))
    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    sorted_oov = sorted(oov, key=oov.get, reverse=True)

    emb_matrix = []
    dest_path = os.path.join(Config.data.base_path, Config.data.processed_path, 'vocab')
    oov_size = 0
    with open(dest_path, 'w', encoding='utf-8') as f:
        f.write(('<pad>' + '\n'))
        f.write(('<unk>' + '\n'))
        f.write(('<s>' + '\n'))
        f.write(('<\s>' + '\n'))
        index = 4
        for word in tqdm(sorted_oov):
            if oov[word] < Config.data.oov_threshold:
                break
            f.write((word + '\n'))
            index += 1
            oov_size += 1
        for word in tqdm(sorted_vocab):
            if vocab[word] < Config.data.word_threshold:
                break
            f.write((word + '\n'))
            emb_matrix.append(embed[word])
            index += 1

    with open(os.path.join(Config.data.base_path, Config.data.processed_path, 'oov_size'), 'w', encoding='utf-8') as f:
        f.write(str(oov_size))

    with open(os.path.join(Config.data.base_path, Config.data.processed_path, 'embed.pkl'), 'wb') as f:
        cPickle.dump(emb_matrix, f)



def load_vocab(vocab_fname):
    print("load vocab ...")
    with open(os.path.join(Config.data.base_path, Config.data.processed_path, vocab_fname), 'r', encoding='utf-8') as f:
        words = f.read().splitlines()
        print("vocab size:", len(words))
    c = Counter(words)
    return {words[i]: i for i in range(len(words))}


def sentence2id(vocab, line):
    # return [vocab.get(token, vocab['<unk>']) for token in segmentor.segment(line)]
    return [vocab.get(token, vocab['<unk>']) for token in line.strip().split('\t')]

def token2id(data, mode):
    """ Convert all the tokens in the data into their corresponding
    index in the vocabulary. """
    vocab_path = 'vocab'
    in_path = data + '.' + mode
    out_path = data + '_ids.' + mode

    vocab = load_vocab(vocab_path)
    in_file = open(os.path.join(Config.data.base_path, Config.data.processed_path, in_path), 'r', encoding='utf-8')
    out_file = open(os.path.join(Config.data.base_path, Config.data.processed_path, out_path), 'w', encoding='utf-8')

    lines = in_file.read().splitlines()
    for line in tqdm(lines):
        if mode == 'dec':  # we only care about '<s>' and </s> in decoder
            ids = [vocab['<s>']]
        else:
            ids = []

        if mode == 'dec':
            sentence_ids = sentence2id(vocab, line)
            ids.extend(sentence_ids)
        else:
            us = json.loads(line)
            for u in us:
                sentence_ids = sentence2id(vocab, u)
                ids.append(sentence_ids)
        if mode == 'dec':
            ids.append(vocab['<\s>'])

        if mode == 'dec':
            out_file.write(' '.join(str(id_) for id_ in ids) + '\n')
        else:
            out_file.write(json.dumps(ids) + '\n')


def prepare_raw_data():
    print('Preparing raw data into train set and test set ...')

    questions, answers = get_question_answers()

    prepare_dataset(questions, answers)

def process_data():
    print('Preparing data to be model-ready ...')

    build_vocab('train.enc', 'train.dec')

    token2id('train', 'enc')
    token2id('train', 'dec')
    token2id('test', 'enc')
    token2id('test', 'dec')


def make_train_and_test_set(shuffle=True, bucket=True, tsize=None):
    """
    :param shuffle: shuffle data
    :param bucket: sort data by length of q & a
    :return: train and test data in form of np array [DATA_SIZE, SENT_SIZE]
    """
    print("make Training data and Test data Start....")

    train_X, train_y = load_data('train_ids.enc', 'train_ids.dec', tsize=tsize) # numpy array, [DATA_SIZE, SNET_SIZE]
    test_X, test_y = load_data('test_ids.enc', 'test_ids.dec')

    assert len(train_X) == len(train_y)
    assert len(test_X) == len(test_y)

    print("train data count :", len(train_X))
    print("test data count :", len(test_X))

    if shuffle:
        print("shuffle dataset ...")
        train_p = np.random.permutation(len(train_y))
        test_p = np.random.permutation(len(test_y))

        train_X, train_y = train_X[train_p], train_y[train_p]
        test_X, test_y = test_X[test_p], test_y[test_p]

    return train_X, test_X, train_y, test_y

def make_eval_set():
    print("make Eval data Start....")

    test_X, test_y = load_data('test_ids.enc', 'test_ids.dec')

    assert len(test_X) == len(test_y)

    print("test data count :", len(test_X))

    return test_X, test_y


def load_data(enc_fname, dec_fname, tsize=None):
    """
    忽略超过长度限制的输入
    忽略问题答案长度超过diff的
    对长度不足的添加pad符号
    :return: numpy array of data
    """
    enc_input_data = open(os.path.join(Config.data.base_path, Config.data.processed_path, enc_fname), 'r', encoding='utf-8')
    dec_input_data = open(os.path.join(Config.data.base_path, Config.data.processed_path, dec_fname), 'r', encoding='utf-8')

    enc_data, dec_data = [], []

    num = 0
    for e_line, d_line in tqdm(zip(enc_input_data.readlines(), dec_input_data.readlines())):
        e_ids = json.loads(e_line)
        d_ids = [int(id_) for id_ in d_line.split()]

        if len(e_ids) == 0 or len(d_ids) == 0:
            continue
        if len(e_ids) > 20:
            continue

        seq_length = max([len(arr) for arr in e_ids])
        if seq_length <= Config.data.max_seq_length and len(d_ids) < Config.data.max_seq_length:
            if abs(len(d_ids) - seq_length) / (seq_length + len(d_ids)) < Config.data.sentence_diff:
                enc_data.append(e_ids)
                dec_data.append(d_ids)
                num += 1
                if tsize is not None and num >= tsize:
                    break

    print("load data from {}, {}, size: {}...".format(enc_fname, dec_fname, num))
    return np.array(enc_data), np.array(dec_data)


def _pad_input(input_, size):
    return input_ + [Config.data.PAD_ID] * (size - len(input_))


def set_max_seq_length(dataset_fnames):

    max_seq_length = Config.data.get('max_seq_length', 10)

    for fname in dataset_fnames:
        input_data = open(os.path.join(Config.data.base_path, Config.data.processed_path, fname), 'r', encoding='utf-8')

        for line in input_data.readlines():
            ids = [int(id_) for id_ in line.split()]
            seq_length = len(ids)

            if seq_length > max_seq_length:
                max_seq_length = seq_length

    Config.data.max_seq_length = max_seq_length
    print("Setting max_seq_length to Config : {max_seq_length}")

#
# def make_batch(data, buffer_size=10000, batch_size=64, scope="train"):
#
#     class IteratorInitializerHook(tf.train.SessionRunHook):
#         """Hook to initialise data iterator after Session is created."""
#
#         def __init__(self):
#             super(IteratorInitializerHook, self).__init__()
#             self.iterator_initializer_func = None
#
#         def after_create_session(self, session, coord):
#             """Initialise the iterator after the session has been created."""
#             self.iterator_initializer_func(session)
#
#
#     def get_inputs():
#
#         iterator_initializer_hook = IteratorInitializerHook()
#
#         def train_inputs():
#             with tf.name_scope(scope):
#
#                 X, y = data
#
#                 # Define placeholders
#                 input_placeholder = tf.placeholder(
#                     tf.int32, [None, Config.data.max_seq_length])
#                 output_placeholder = tf.placeholder(
#                     tf.int32, [None, Config.data.max_seq_length])
#
#                 # Build dataset iterator
#                 # Creates a Dataset whose elements are slices of the given tensors.
#                 dataset = tf.data.Dataset.from_tensor_slices(
#                     (input_placeholder, output_placeholder))
#
#                 if scope == "train":
#                     dataset = dataset.repeat(None)  # Infinite iterations
#                 else:
#                     dataset = dataset.repeat(1)  # 1 Epoch
#                 # dataset = dataset.shuffle(buffer_size=buffer_size)
#                 dataset = dataset.batch(batch_size)
#
#                 iterator = dataset.make_initializable_iterator()
#                 next_X, next_y = iterator.get_next()
#
#                 tf.identity(next_X[0], 'enc_0')
#                 tf.identity(next_y[0], 'dec_0')
#
#                 # Set runhook to initialize iterator
#                 iterator_initializer_hook.iterator_initializer_func = \
#                     lambda sess: sess.run(
#                         iterator.initializer,
#                         feed_dict={input_placeholder: X,
#                                    output_placeholder: y})
#
#                 # Return batched (features, labels)
#                 return next_X, next_y
#
#         # Return function and hook
#         return train_inputs, iterator_initializer_hook
#
#     return get_inputs()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)

    prepare_raw_data()
    process_data()
