#-- coding: utf-8 -*-

import argparse
import atexit
import logging
import os
#
from hbconfig import Config
import data_loader
import utils
import  time
from HRAN import Model

import dynet as dy
from nltk.translate.bleu_score import sentence_bleu

# model = dy.Model()
# input_lookup = model.add_lookup_parameters((10, 128))
# print(input_lookup[1].value())

def blue_score(gold, pred):
    gold = [gold]
    score = sentence_bleu(gold, pred, [0.25, 0.25, 0.25, 0.25])
    return score

def eval(x, y, model):
    def to_str(sequence):
        tokens = [
            Config.data.rev_vocab.get(x, '') for x in sequence if x != Config.data.PAD_ID]
        return ' '.join(tokens)
    total_blue = 0
    pred_y = []
    for i in range(len(x)):
        pred = model.predict(x[i])
        pred_y.append(pred)
    with open(os.path.join(Config.model_dir, 'pred.txt'), 'w', encoding='utf-8') as f:
        for i in range(len(x)):
            for u in x[i]:
                f.write(to_str(u) + '\n')
            gold = to_str(y[i])
            p = to_str(pred[i])
            blue = blue_score(gold, p)
            f.write('正确> ' + gold + '\n')
            f.write('预测> ' + p + '\n')
            f.write('score> ' + str(blue) + '\n')
            total_blue += blue

    with open(os.path.join(Config.model_dir, 'eval.data'), 'w', encoding='utf-8') as f:
        f.write('blue score: ' + str(total_blue / len(x)))


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
    losses =[]
    dy.renew_cg()
    for i in range(len(x)):
        loss = model.train(x[i], y[i])
        # loss_value = loss.value()
        # iter_loss += loss_value
        # total_loss += loss_value
        # loss.backward()
        # trainer.update()
        losses.append(loss)
        if len(losses) == Config.model.batch_size:
            loss = dy.esum(losses)
            loss_value = loss.value()
            loss.backward()
            trainer.update()
            iter_loss += loss_value
            total_loss += loss_value
            losses = []
            dy.renew_cg()
        # if (i + 1) % (Config.train.loss_hook_n_iter * Config.model.batch_size) == 0:# and i != 0:
        #     print('Step {}/{}, loss {:.2f}, time {:.2f}s'.format(i, len(x), iter_loss / (Config.train.loss_hook_n_iter * Config.model.batch_size), time.time() - start_iter))
        #     iter_loss = 0
        #     start_iter = time.time()
        #     pred = model.predict(tx)
        #     for ids in tx:
        #         print(to_str(ids))
        #     print('预测>', to_str(pred))
        #     print('正确>', to_str(ty))
    if len(losses) != 0:
        loss = dy.esum(losses)
        loss.backward()
        trainer.update()
        loss_value = loss.value()
        total_loss += loss_value
        dy.renew_cg()
    print('Loss {:.2f}, time {:.2f}s'.format(total_loss / len(x), time.time() - start))


def main(Config):
    # 返回字典
    vocab = data_loader.load_vocab("vocab")
    Config.data.vocab_size = len(vocab)
    Config.data.vocab = vocab
    rev_vocab = utils.get_rev_vocab(vocab)
    Config.data.rev_vocab = rev_vocab

    # 定义训练数据
    train_X, test_X, train_y, test_y = data_loader.make_train_and_test_set()

    model = Model(Config)

    for e in range(Config.train.epoch):
        train(train_X[: 10], train_y[: 10], model, Config, train_X[0], train_y[0])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config',
                        help='config file name')
    parser.add_argument('--mode', type=str, default='train',
                        help='Mode (train/test/train_and_evaluate)')
    parser.add_argument('--dynet-autobatch', type=int, default=1)
    parser.add_argument('--dynet-mem', type=int, default=2048)
    args = parser.parse_args()

    Config(args.config)
    print("Config: ", Config)
    if Config.get("description", None):
        print("Config Description")
        for key, value in Config.description.items():
            print(" - {}: {}".format(key, value))

    main(Config)
