import dynet as dy
import random
import numpy as np
import heapq
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
        ATTENTION_SIZE = Config.model.attention_size

        with open(os.path.join(Config.data.base_path, Config.data.processed_path, 'embed.pkl'), 'rb') as f:
            embed = np.asarray(cPickle.load(f))
        oov = np.random.random((4 + Config.data.oov_size, EMBEDDINGS_SIZE))

        self.input_lookup = self.model.lookup_parameters_from_numpy(np.concatenate((oov, embed)))
        self.enc_fwd_lstm = dy.GRUBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, self.model)
        self.enc_bwd_lstm = dy.GRUBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, self.model)
        self.attention_word_w1 = self.model.add_parameters((ATTENTION_SIZE, STATE_SIZE * 2), init='uniform')
        self.attention_word_w2 = self.model.add_parameters((ATTENTION_SIZE, STATE_SIZE * LSTM_NUM_OF_LAYERS * 1), init='uniform')
        self.attention_word_w3 = self.model.add_parameters((ATTENTION_SIZE, STATE_SIZE * LSTM_NUM_OF_LAYERS * 1), init='uniform')
        self.attention_word_v = self.model.add_parameters((1, ATTENTION_SIZE), init='uniform')
        self.utt_lstm = dy.GRUBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE * 2, STATE_SIZE, self.model)
        self.attention_utt_w1 = self.model.add_parameters((ATTENTION_SIZE, STATE_SIZE), init='uniform')
        self.attention_utt_w2 = self.model.add_parameters((ATTENTION_SIZE, STATE_SIZE * LSTM_NUM_OF_LAYERS * 1), init='uniform')
        self.attention_utt_v = self.model.add_parameters((1, ATTENTION_SIZE), init='uniform')
        self.sess_lstm = dy.GRUBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE + EMBEDDINGS_SIZE, STATE_SIZE, self.model)
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

    def get_word_att(self, ut, l, s):
        input_mat = dy.concatenate_cols(ut.words_enc)

        unnormalized = dy.transpose(self.attention_word_v * dy.tanh(dy.colwise_add(dy.colwise_add(self.attention_word_w1 * input_mat, self.attention_word_w2 * l), self.attention_word_w3 *s)))
        att_weights = dy.softmax(unnormalized)

        ut.context = input_mat * att_weights

    def get_utt_att(self, uts, s):
        input_mat = dy.concatenate_cols([u.utt_enc for u in uts])

        unnormalized = dy.transpose(self.attention_word_v * dy.tanh(dy.colwise_add(self.attention_utt_w1 * input_mat, self.attention_utt_w2 * s)))
        att_weights = dy.softmax(unnormalized)

        return input_mat * att_weights

    def run_lstm(self, init_state, input_vecs):
        s = init_state

        out_vectors = []
        for vector in input_vecs:
            s = s.add_input(vector)
            out_vector = s.output()
            out_vectors.append(out_vector)
        return out_vectors

    def encode_words(self, ut):
        sentence_rev = list(reversed(ut.words_emb))

        fwd_vectors = self.run_lstm(self.enc_fwd_lstm.initial_state(), ut.words_emb)
        bwd_vectors = self.run_lstm(self.enc_bwd_lstm.initial_state(), sentence_rev)
        bwd_vectors = list(reversed(bwd_vectors))
        ut.words_enc = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

    def train(self, inputs, target):
        dropout = self.Config.train.dropout
        uts = []
        for u in inputs:
            u = Utt(u)
            u.words_emb = []
            for word in u.words:
                u.words_emb.append(dy.dropout(dy.lookup(self.input_lookup, word, update=True if word < 4 + self.Config.data.oov_size else False), dropout))
            self.encode_words(u)
            uts.append(u)

        # last_output_embeddings = self.lookup(self.Config.data.START_ID, emb)
        last_output_embeddings = self.input_lookup[self.Config.data.START_ID]
        s = self.sess_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(self.Config.model.num_units), last_output_embeddings]))
        loss = []

        for gt in target:
            spt = dy.concatenate(list(s.s()))
            l = self.utt_lstm.initial_state_from_raw_vectors(
                [np.random.normal(0, 0.1, self.Config.model.num_units) for i in range(1 * self.Config.model.num_layers)])
            lpt = dy.concatenate(list(l.s()))

            # encode utt
            for i in range(len(uts) - 1, -1, -1):
                self.get_word_att(uts[i], lpt, spt)
                l = l.add_input(uts[i].context)
                uts[i].utt_enc = l.output()
                lpt = dy.concatenate(list(l.s()))

            # decode
            c = self.get_utt_att(uts, spt)
            s = s.add_input(dy.concatenate([c, last_output_embeddings]))
            probs = dy.softmax(self.decoder_w * s.output() + self.decoder_b)
            # last_output_embeddings = self.lookup(gt, emb)
            last_output_embeddings = dy.lookup(self.input_lookup, gt, update=True if gt < 4 + self.Config.data.oov_size else False)
            loss.append(-dy.log(dy.pick(probs, gt)))

        loss = dy.esum(loss)
        return loss

    def attend(self, input_mat, state, w1dt):
        w2 = dy.parameter(self.attention_utt_w2)
        v = dy.parameter(self.attention_utt_v)

        # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
        # w1dt: (attdim x seqlen)
        # w2dt: (attdim x attdim)
        w2dt = w2 * dy.concatenate(list(state.s()))
        # att_weights: (seqlen,) row vector
        unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
        att_weights = dy.softmax(unnormalized)
        # context: (encoder_state)
        context = input_mat * att_weights
        return context

    def train2(self, inputs, target):
        # oneline = []
        # for input in inputs:
        #     oneline.extend(input)
        # emb = dy.const_parameter(self.embed)
        oneline = inputs
        words_emb = []
        for word in oneline:
            # words_emb.append(self.lookup(word, emb))
            words_emb.append(self.input_lookup[word])
        sentence_rev = list(reversed(words_emb))

        fwd_vectors = self.run_lstm(self.enc_fwd_lstm.initial_state(), words_emb)
        bwd_vectors = self.run_lstm(self.enc_bwd_lstm.initial_state(), sentence_rev)
        bwd_vectors = list(reversed(bwd_vectors))
        words_enc = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

        w = dy.parameter(self.decoder_w)
        b = dy.parameter(self.decoder_b)

        # last_output_embeddings = self.input_lookup[self.Config.data.START_ID]
        # s = self.sess_lstm.initial_state().add_input(
        #     dy.concatenate([dy.vecInput(self.Config.model.num_units * 2), last_output_embeddings]))
        # loss = []
        #
        # for gt in target:
        #     spt = dy.concatenate(list(s.s()))
        #     c = self.get_utt_att(words_enc, spt)
        #     s.add_input(dy.concatenate([c, last_output_embeddings]))
        #     probs = dy.softmax(w * s.output() + b)
        #     last_output_embeddings = self.input_lookup[gt]
        #     loss.append(-dy.log(dy.pick(probs, gt)))
        #
        # return dy.esum(loss)
        w1 = dy.parameter(self.attention_utt_w1)
        input_mat = dy.concatenate_cols(words_enc)
        w1dt = None

        # last_output_embeddings = self.lookup(self.Config.data.START_ID, emb)
        last_output_embeddings = self.input_lookup[self.Config.data.START_ID]
        s = self.sess_lstm.initial_state().add_input(
            dy.concatenate([dy.vecInput(self.Config.model.num_units * 2), last_output_embeddings]))
        loss = []

        for char in target:
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or w1 * input_mat
            vector = dy.concatenate([self.attend(input_mat, s, w1dt), last_output_embeddings])
            s = s.add_input(vector)
            out_vector = w * s.output() + b
            probs = dy.softmax(out_vector)
            # last_output_embeddings = self.lookup(char, emb)
            last_output_embeddings = self.input_lookup[char]
            loss.append(-dy.log(dy.pick(probs, char)))
        loss = dy.esum(loss)

        return loss

    def predict(self, inputs):
        beam = self.Config.predict.beam_width
        uts = []
        for u in inputs:
            u = Utt(u)
            u.words_emb = []
            for word in u.words:
                u.words_emb.append(
                    dy.lookup(self.input_lookup, word, update=True if word < 4 + self.Config.data.oov_size else False))
            self.encode_words(u)
            uts.append(u)

        w = dy.parameter(self.decoder_w)
        b = dy.parameter(self.decoder_b)

        # last_output_embeddings = self.lookup(self.Config.data.START_ID, self.embed)
        last_output_embeddings = self.input_lookup[self.Config.data.START_ID]
        s = self.sess_lstm.initial_state().add_input(
            dy.concatenate([dy.vecInput(self.Config.model.num_units), last_output_embeddings]))

        out = []
        seq_len = max([len(x) for x in inputs])
        for i in range(seq_len * 2):
            spt = dy.concatenate(list(s.s()))
            l = self.utt_lstm.initial_state_from_raw_vectors(
                [np.random.normal(0, 0.1, self.Config.model.num_units) for i in
                 range(1 * self.Config.model.num_layers)])
            lpt = dy.concatenate(list(l.s()))

            # encode utt
            for i in range(len(uts) - 1, -1, -1):
                self.get_word_att(uts[i], lpt, spt)
                l = l.add_input(uts[i].context)
                uts[i].utt_enc = l.output()
                lpt = dy.concatenate(list(l.s()))

            # decode
            c = self.get_utt_att(uts, spt)
            s = s.add_input(dy.concatenate([c, last_output_embeddings]))
            probs = dy.softmax(w * s.output() + b).vec_value()
            next_word = probs.index(max(probs))
            # val, idx = probs.topk()
            # last_output_embeddings = self.lookup(next_word, self.embed)
            last_output_embeddings = self.input_lookup[next_word]
            if next_word == self.Config.data.EOS_ID:
                break
            if next_word != self.Config.data.START_ID:
                out.append(next_word)
        return out

    def beam_pred(self, inputs):
        beam = self.Config.predict.beam_width
        uts = []
        for u in inputs:
            u = Utt(u)
            u.words_emb = []
            for word in u.words:
                u.words_emb.append(
                    dy.lookup(self.input_lookup, word, update=True if word < 4 + self.Config.data.oov_size else False))
            self.encode_words(u)
            uts.append(u)

        w = dy.parameter(self.decoder_w)
        b = dy.parameter(self.decoder_b)

        # last_output_embeddings = self.lookup(self.Config.data.START_ID, self.embed)
        last_output_embeddings = self.input_lookup[self.Config.data.START_ID]
        s = self.sess_lstm.initial_state().add_input(
            dy.concatenate([dy.vecInput(self.Config.model.num_units), last_output_embeddings]))

        seq_len = max([len(x) for x in inputs])
        top_k = {'': [0, self.Config.data.START_ID, s]}
        for i in range(seq_len * 2):
            new_top = {}
            fini = 0
            for item in top_k:
                if top_k[item][1] == self.Config.data.EOS_ID:
                    new_top[item] = top_k[item]
                    fini += 1
                    continue
                last_output_embeddings = self.input_lookup[top_k[item][1]]
                s = top_k[item][2]

                spt = dy.concatenate(list(s.s()))
                l = self.utt_lstm.initial_state_from_raw_vectors(
                    [np.random.normal(0, 0.1, self.Config.model.num_units) for i in
                     range(1 * self.Config.model.num_layers)])
                lpt = dy.concatenate(list(l.s()))

                # encode utt
                for i in range(len(uts) - 1, -1, -1):
                    self.get_word_att(uts[i], lpt, spt)
                    l = l.add_input(uts[i].context)
                    uts[i].utt_enc = l.output()
                    lpt = dy.concatenate(list(l.s()))

                # decode
                c = self.get_utt_att(uts, spt)
                s = s.add_input(dy.concatenate([c, last_output_embeddings]))
                probs = dy.log(dy.softmax(w * s.output() + b)).vec_value()
                index, val = zip(*heapq.nlargest(beam, enumerate(probs), key=lambda probs: probs[1]))
                if index[0] == self.Config.data.START_ID:
                    new_top[item] = [0, self.Config.data.START_ID, s]
                else:
                    for rk in range(len(index)):
                        if index[rk] == self.Config.data.EOS_ID:
                            new_top[item] = [top_k[item][0] + val[rk] - rk, index[rk], s]
                        elif not item + self.Config.data.rev_vocab.get(index[rk], "") in new_top:
                            new_top[item + self.Config.data.rev_vocab.get(index[rk], "")] = [top_k[item][0] + val[rk] - rk, index[rk], s]
            top_k = {x: new_top[x] for x in heapq.nlargest(beam, new_top, key=lambda item: new_top[item][0])}
            if fini == beam:
                break
        return list(top_k.keys())
