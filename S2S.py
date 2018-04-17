import dynet as dy
import random
import numpy as np
import os
from six.moves import cPickle

class Utt:
    def __init__(self, raw):
        self.words = raw
        self.words_emb = None
        self.words_enc = None
        self.context = None
        self.utt_enc = None

class Model:
    def __init__(self, Config):
        self.Config = Config
        self.model = dy.Model()

        VOCAB_SIZE = Config.data.vocab_size
        EMBEDDINGS_SIZE = Config.model.embed_dim
        LSTM_NUM_OF_LAYERS = Config.model.num_layers
        STATE_SIZE = Config.model.num_units

        with open(os.path.join(Config.data.base_path, Config.data.processed_path, 'embed.pkl'), 'rb') as f:
            embed = np.asarray(cPickle.load(f))
        oov = np.random.random((4 + Config.data.oov_size, EMBEDDINGS_SIZE))

        # self.embed = self.model.lookup_parameters_from_numpy(np.transpose(np.asarray(embed)))
        # self.oov = self.model.add_lookup_parameters((4 + Config.data.oov_size, EMBEDDINGS_SIZE), init='uniform')
        self.input_lookup = self.model.lookup_parameters_from_numpy(np.concatenate((oov, embed)))
        self.enc_fwd_lstm = dy.GRUBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, self.model)
        self.enc_bwd_lstm = dy.GRUBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, self.model)
        self.sess_lstm = dy.GRUBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE, STATE_SIZE, self.model)
        self.decoder_w = self.model.add_parameters((VOCAB_SIZE, STATE_SIZE), init='uniform')
        self.decoder_b = self.model.add_parameters((VOCAB_SIZE), init='uniform')

    def save(self, name='model'):
        if not os.path.exists(self.Config.train.model_dir):
            os.makedirs(self.Config.train.model_dir)
        save_path = os.path.join(self.Config.train.model_dir, name)
        self.model.save(save_path)

    def load(self, name='model'):
        load_path = os.path.join(self.Config.train.model_dir, name)
        if not os.path.exists(load_path):
            print("Path {} not found".format(load_path))
        self.model.populate(load_path)

    def embed_words(self, ut, emb):
        ut.words_emb = []
        for word in ut.words:
            ut.words_emb.append(dy.lookup(self.input_lookup, word, update=True if word < 4 + self.Config.data.oov_size else False))
            # ut.words_emb.append(self.input_lookup[word])

    def run_lstm(self, init_state, input_vecs):
        s = init_state

        out_vectors = []
        for vector in input_vecs:
            s = s.add_input(vector)
            out_vector = s.output()
            out_vectors.append(out_vector)
        return out_vectors, s

    def train(self, inputs, target):
        words_emb = []
        dropout = self.Config.train.dropout
        for u in inputs:
            for word in u:
                words_emb.append(dy.dropout(dy.lookup(self.input_lookup, word, update=True if word < 4 + self.Config.data.oov_size else False), dropout))
                # words_emb.append(self.input_lookup[word])
        fwd_vectors, state = self.run_lstm(self.enc_fwd_lstm.initial_state(), words_emb)

        # s = self.sess_lstm.initial_state(state.s()).add_input(dy.lookup(self.input_lookup, self.Config.data.EOS_ID))
        s = self.sess_lstm.initial_state(state.s()).add_input(self.input_lookup[self.Config.data.EOS_ID])
        loss = []

        for char in target:
            s = s.add_input(s.output())
            out_vector = self.decoder_w * s.output() + self.decoder_b
            probs = dy.softmax(out_vector)
            loss.append(-dy.log(dy.pick(probs, char)))
        loss = dy.esum(loss)

        return loss

    def predict(self, inputs):
        words_emb = []
        for u in inputs:
            for word in u:
                words_emb.append(self.input_lookup[word])


        fwd_vectors, state = self.run_lstm(self.enc_fwd_lstm.initial_state(), words_emb)


        # s = self.sess_lstm.initial_state().add_input(fwd_vectors[-1])
        s = self.sess_lstm.initial_state(state.s()).add_input(self.input_lookup[self.Config.data.EOS_ID])

        out = []
        seq_len = max([len(x) for x in inputs])
        for i in range(seq_len * 2):
            s = s.add_input(s.output())

            probs = dy.softmax(self.decoder_w * s.output() + self.decoder_b).vec_value()
            next_word = probs.index(max(probs))
            if next_word == self.Config.data.EOS_ID:
                break
            if next_word != self.Config.data.START_ID:
                out.append(next_word)
        return out
