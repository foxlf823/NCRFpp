# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-02-13 11:49:38

from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wordsequence2 import WordSequence2
from .crf import CRF

class SeqLabel2(nn.Module):
    def __init__(self, data, target=True):
        super(SeqLabel2, self).__init__()
        self.use_crf = data.use_crf
        print("build sequence labeling network...")
        print("use_char: ", data.use_char)
        if data.use_char:
            print("char feature extractor: ", data.char_feature_extractor)
        print("word feature extractor: ", data.word_feature_extractor)
        print("use crf: ", self.use_crf)

        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss
        ## add two more label for downlayer lstm, use original label size for CRF
        if target:
            label_size = data.label_alphabet_size
            data.label_alphabet_size += 2
        else:
            label_size = data.s_label_alphabet_size
            data.s_label_alphabet_size += 2

        self.word_hidden = WordSequence2(data)

        # The linear layer that maps from hidden state space to tag space
        self.use_elmo = data.use_elmo
        if self.use_elmo:
            self.hidden2tag = nn.Linear(data.HP_hidden_dim // 2, data.label_alphabet_size if target else data.s_label_alphabet_size)
        else:
            self.hidden2tag = nn.Linear(data.HP_hidden_dim, data.label_alphabet_size if target else data.s_label_alphabet_size)

        if self.gpu >= 0 and torch.cuda.is_available():
            self.hidden2tag = self.hidden2tag.cuda(self.gpu)

        if self.use_crf:
            self.crf = CRF(label_size, self.gpu)


    def calculate_loss(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask,
                       elmo_char_inputs, target=True):
        outs = self.word_hidden(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, elmo_char_inputs, target)
        ## outs (batch_size, seq_len, hidden_size)
        outs = self.hidden2tag(outs)

        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
            scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        else:
            loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
            outs = outs.view(batch_size * seq_len, -1)
            score = F.log_softmax(outs, 1)
            total_loss = loss_function(score, batch_label.view(batch_size * seq_len))
            _, tag_seq  = torch.max(score, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
        if self.average_batch:
            total_loss = total_loss / batch_size
        return total_loss, tag_seq


    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask,
                elmo_char_inputs, target=True):
        outs = self.word_hidden(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, elmo_char_inputs, target)
        ## outs (batch_size, seq_len, hidden_size)
        outs = self.hidden2tag(outs)

        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        if self.use_crf:
            scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        else:
            outs = outs.view(batch_size * seq_len, -1)
            _, tag_seq  = torch.max(outs, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            ## filter padded position with zero
            tag_seq = mask.long() * tag_seq
        return tag_seq


    # def get_lstm_features(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
    #     return self.word_hidden(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)


    def decode_nbest(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask, nbest,
                     elmo_char_inputs):
        raise RuntimeError('not support')