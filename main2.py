#-- coding: utf-8 -*-

import argparse
import atexit
import logging
import os
#
import gc
import os
from hbconfig import Config
import data_loader
import utils
import  time
from HRAN import Model
# from S2S import Model
from six.moves import cPickle
import shutil

import dynet as dy
from nltk.translate.bleu_score import sentence_bleu

# model = dy.Model()
# input_lookup = model.add_lookup_parameters((10, 128))
# print(input_lookup[1].value())
best_blue = -1


def blue_score(gold, pred):
    gold = [gold]
    score = sentence_bleu(gold, pred, [0.25, 0.25, 0.25, 0.25])
    return score

def eval(x, y, model):
    def to_str(sequence):
        tokens = [
            Config.data.rev_vocab.get(x, '') for x in sequence if (x > 3)]
        return ''.join(tokens)
    total_blue = 0
    with open(os.path.join(Config.train.model_dir, 'pred.txt'), 'w', encoding='utf-8') as f:
        for i in range(len(x)):
            if Config.predict.beam_width < 2:
                pred = model.predict(x[i])
                for u in x[i]:
                    f.write(to_str(u) + '\n')
                gold = to_str(y[i])
                p = to_str(pred)
                blue = blue_score(gold, p)
                f.write('正确> ' + gold + '\n')
                f.write('预测> ' + p + '\n')
                f.write('score> ' + str(blue) + '\n')
                total_blue += blue
            else:
                pred = model.beam_pred(x[i])
                for u in x[i]:
                    f.write(to_str(u) + '\n')
                gold = to_str(y[i])
                blue = blue_score(gold, pred[0])
                f.write('正确> ' + gold + '\n')
                f.write('预测> ' + '\n    '.join(pred) + '\n')
                f.write('score> ' + str(blue) + '\n')
                total_blue += blue

    with open(os.path.join(Config.train.model_dir, 'eval.data'), 'w', encoding='utf-8') as f:
        f.write('blue score: ' + str(total_blue / len(x) * 100))


def get_data_blue(x, y, model):
    def to_str(sequence):
        tokens = [
            Config.data.rev_vocab.get(x, '') for x in sequence if (x > 3)]
        return ''.join(tokens)
    total_blue = 0
    for i in range(len(x)):
        dy.renew_cg()
        pred = model.predict(x[i])
        gold = to_str(y[i])
        p = to_str(pred)
        blue = blue_score(gold, p)
        total_blue += blue
    return total_blue / len(x)

def train(x, y, model, Config, dev_x, dev_y, e, trainer):
    start = time.time()
    start_iter = time.time()
    total_loss = 0
    iter_loss = 0
    losses =[]
    dy.renew_cg()
    global best_blue
    for i in range(len(x)):
        loss = model.train(x[i], y[i])
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
        if (i + 1) % (Config.train.loss_hook_n_iter * Config.model.batch_size) == 0:# and i != 0:
            print('Step {}/{}, loss {:.2f}, time {:.2f}s'.format(i + 1, len(x), iter_loss / (Config.train.loss_hook_n_iter * Config.model.batch_size), time.time() - start_iter))
            iter_loss = 0
            start_iter = time.time()
        if (i + 1) % (Config.train.save_checkpoints_steps * Config.model.batch_size) == 0:
            blue = get_data_blue(dev_x, dev_y, model)
            print('Checkpoint {}-{}: loss {:.2f}, time {:.2f}s, dev_blue {:.2f}'.format(e + 1, i + 1, total_loss / (i + 1),
                                                                                time.time() - start, blue * 100))
            if blue > best_blue:
                model.save('model-{}-{}'.format(e+1, i+1))
    if len(losses) != 0:
        loss = dy.esum(losses)
        loss.backward()
        trainer.update()
        loss_value = loss.value()
        total_loss += loss_value
        dy.renew_cg()
    blue = get_data_blue(dev_x, dev_y, model)
    # blue = 0
    print('Epoch {}: loss {:.2f}, time {:.2f}s, dev_blue {:.2f}'.format(e + 1, total_loss / len(x), time.time() - start, blue * 100))
    return blue

def main(Config, mode):
    # 返回字典
    vocab = data_loader.load_vocab("vocab")
    Config.data.vocab_size = len(vocab)
    with open(os.path.join(Config.data.base_path, Config.data.processed_path, 'oov_size'), 'r', encoding='utf-8') as f:
        oov_size = int(f.readline().strip())
    Config.data.oov_size = oov_size
    Config.data.vocab = vocab
    rev_vocab = utils.get_rev_vocab(vocab)
    Config.data.rev_vocab = rev_vocab
    if mode == 'train':
        # save_path = os.path.join(Config.train.model_dir)
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # with open(os.path.join(save_path, 'vocab.pkl'), 'wb') as f:
        #     cPickle.dump(vocab, f)

        # 定义训练数据
        train_X, test_X, train_y, test_y = data_loader.make_train_and_test_set(tsize=Config.train.size)

        model = Model(Config)
        trainer = dy.AdamTrainer(model.model)
        # model.load('model-1-final')
        global best_blue

        for e in range(Config.train.epoch):
            dev_blue = train(train_X, train_y, model, Config, test_X, test_y, e, trainer)
            if dev_blue > best_blue:
            # if (e + 1) % 50 == 0:
                best_blue = dev_blue
                model.save('model-{}-{}'.format(e + 1, 'final'))
                eval(train_X, train_y, model)

    if mode == 'eval':
        # save_path = os.path.join(Config.train.model_dir)
        # with open(os.path.join(save_path, 'vocab.pkl'), 'rb') as f:
        #     vocab = cPickle.load(f)
        Config.vocab = vocab
        rev_vocab = utils.get_rev_vocab(vocab)
        Config.data.rev_vocab = rev_vocab

        test_X, test_y = data_loader.make_eval_set()

        model = Model(Config)
        model.load('model-1-final')

        eval(test_X, test_y, model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config',
                        help='config file name')
    parser.add_argument('--mode', type=str, default='train',
                        help='Mode (train/test/train_and_evaluate)')
    parser.add_argument('--dynet-autobatch', type=int, default=1)
    parser.add_argument('--dynet-mem', type=int, default=2048)
    parser.add_argument('--dynet-gpu', type=int, default=1)
    args = parser.parse_args()

    Config(args.config)
    print("Config: ", Config)
    # if Config.get("description", None):
    #     print("Config Description")
    #     for key, value in Config.description.items():
    #         print(" - {}: {}".format(key, value))

    main(Config, 'train')
