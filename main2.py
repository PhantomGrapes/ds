#-- coding: utf-8 -*-

import argparse
import atexit
import logging
#
import os
from hbconfig import Config
import data_loader
import hook
import utils
import  time
from HRAN import Model
from six.moves import cPickle
import shutil

import dynet as dy

# model = dy.Model()
# input_lookup = model.add_lookup_parameters((10, 128))
# print(input_lookup[1].value())

def predict(x, model, Config):
    y = []
    for inputs in x:
        y.append(model.predict(inputs))


def train(x, y, model, Config, tx, ty):
    def to_str(sequence):
        tokens = [
            Config.data.rev_vocab.get(x, '') for x in sequence if x != Config.data.PAD_ID]
        return ' '.join(tokens)
    start = time.time()
    start_iter = time.time()
    trainer = dy.AdamTrainer(model.model)
    total_loss = 0
    iter_loss = 0
    for i in range(len(x)):
        dy.renew_cg()
        loss = model.train(x[i], y[i])
        loss_value = loss.value()
        iter_loss += loss_value
        total_loss += loss_value
        loss.backward()
        trainer.update()
        if i % Config.train.loss_hook_n_iter == 0 and i != 0:
            print('Step {}/{}, loss {:.2f}, time {:.2f}s'.format(i, len(x), iter_loss / Config.train.loss_hook_n_iter, time.time() - start_iter))
            iter_loss = 0
            start_iter = time.time()
            pred = model.predict(tx)
            for ids in tx:
                print(to_str(ids))
            print('预测>', to_str(pred))
            print('正确>', to_str(ty))

    print('Loss {:.2f}, time {:.2f}s'.format(total_loss / len(x), time.time() - start))


def main(Config, mode):
    # 返回字典
    vocab = data_loader.load_vocab("vocab")
    Config.data.vocab_size = len(vocab)
    Config.data.vocab = vocab
    rev_vocab = utils.get_rev_vocab(vocab)
    Config.data.rev_vocab = rev_vocab
    if mode == 'train':
        save_path = os.path.join(Config.train.model_dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        shutil.rmtree(save_path)
        with open(os.path.join(save_path, 'config.pkl'), 'wb') as f:
            cPickle.dump(Config.to_dict(), f)

    # 定义训练数据
    train_X, test_X, train_y, test_y = data_loader.make_train_and_test_set()

    model = Model(Config)

    for e in range(Config.train.epoch):
        train(train_X, train_y, model, Config, test_X[0], test_y[0])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config',
                        help='config file name')
    parser.add_argument('--mode', type=str, default='train',
                        help='Mode (train/test/train_and_evaluate)')
    args = parser.parse_args()

    # Print Config setting
    Config(args.config)
    print("Config: ", Config)
    if Config.get("description", None):
        print("Config Description")
        for key, value in Config.description.items():
            print(" - {}: {}".format(key, value))

    main(Config, 'train')