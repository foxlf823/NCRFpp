# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-02-01 15:59:26
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .wordrep import WordRep
import json

class WordSequence1(nn.Module):
    def __init__(self, data):
        super(WordSequence1, self).__init__()
        print("build word sequence feature extractor: %s..."%(data.word_feature_extractor))
        self.gpu = data.HP_gpu
        self.use_char = data.use_char
        # self.batch_size = data.HP_batch_size
        # self.hidden_dim = data.HP_hidden_dim
        self.droplstm = nn.Dropout(data.HP_dropout)
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        self.wordrep = WordRep(data)
        self.input_size = data.word_emb_dim
        self.feature_num = data.feature_num
        if self.use_char:
            self.input_size += data.HP_char_hidden_dim
            if data.char_feature_extractor == "ALL":
                self.input_size += data.HP_char_hidden_dim
        for idx in range(self.feature_num):
            self.input_size += data.feature_emb_dims[idx]
        self.use_elmo = data.use_elmo
        if self.use_elmo:
            with open(data.elmo_options_file, 'r') as fin:
                self._options = json.load(fin)
            self.input_size += self._options['lstm']['projection_dim']*2
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim

        self.word_feature_extractor = data.word_feature_extractor

        if self.use_elmo:
            # self.lstm1 = nn.LSTM(self.input_size, lstm_hidden, num_layers=1, batch_first=True, bidirectional=self.bilstm_flag)
            # self.lstm2 = nn.LSTM(lstm_hidden*2, lstm_hidden // 2, num_layers=1, batch_first=True, bidirectional=self.bilstm_flag)

            self.lstm1_forward = nn.LSTM(self.input_size, lstm_hidden, num_layers=1, batch_first=True, bidirectional=False)
            self.lstm1_backward = nn.LSTM(self.input_size, lstm_hidden, num_layers=1, batch_first=True, bidirectional=False)
            self.lstm2_forward = nn.LSTM(lstm_hidden, lstm_hidden // 2, num_layers=1, batch_first=True, bidirectional=False)
            self.lstm2_backward = nn.LSTM(lstm_hidden, lstm_hidden // 2, num_layers=1, batch_first=True, bidirectional=False)
        else:
            # self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)

            self.lstm_forward = nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=False)
            self.lstm_backward = nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=False)



        if self.gpu >= 0 and torch.cuda.is_available():
            self.droplstm = self.droplstm.cuda(self.gpu)


            if self.use_elmo:
                # self.lstm1 = self.lstm1.cuda(self.gpu)
                # self.lstm2 = self.lstm2.cuda(self.gpu)
                self.lstm1_forward = self.lstm1_forward.cuda(self.gpu)
                self.lstm1_backward = self.lstm1_backward.cuda(self.gpu)
                self.lstm2_forward = self.lstm2_forward.cuda(self.gpu)
                self.lstm2_backward = self.lstm2_backward.cuda(self.gpu)
            else:
                # self.lstm = self.lstm.cuda(self.gpu)
                self.lstm_forward = self.lstm_forward.cuda(self.gpu)
                self.lstm_backward = self.lstm_backward.cuda(self.gpu)


    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, elmo_char_inputs):
        """
            input:
                word_inputs: (batch_size, sent_len)
                feature_inputs: [(batch_size, sent_len), ...] list of variables
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        
        word_represent = self.wordrep(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, elmo_char_inputs)
        ## word_embs (batch_size, seq_len, embed_size)
        packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        if self.use_elmo:
            # lstm_out, hidden = self.lstm1(packed_words, None)
            # lstm_out, hidden = self.lstm2(lstm_out, None)

            lstm_out_forward, hidden_forward = self.lstm1_forward(packed_words, None)
            lstm_out_backward, hidden_backward = self.lstm1_backward(packed_words, None)
            lstm_out_forward, hidden_forward = self.lstm2_forward(lstm_out_forward, None)
            lstm_out_backward, hidden_backward = self.lstm2_backward(lstm_out_backward, None)
        else:
            # lstm_out, hidden = self.lstm(packed_words, hidden)

            lstm_out_forward, hidden_forward = self.lstm_forward(packed_words, None)
            lstm_out_backward, hidden_backward = self.lstm_backward(packed_words, None)
        ## lstm_out_forward (seq_len, batch, hidden_size)
        lstm_out_forward, _ = pad_packed_sequence(lstm_out_forward)
        lstm_out_backward, _ = pad_packed_sequence(lstm_out_backward)


        feature_out_forward = self.droplstm(lstm_out_forward.transpose(1,0))
        feature_out_backward = self.droplstm(lstm_out_backward.transpose(1,0))
        ## feature_out (batch_size, seq_len, hidden_size)

        return feature_out_forward, feature_out_backward

