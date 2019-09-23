

from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.wordsequence import WordSequence
from model.crf import CRF
import json
from .elmo import Elmo

class Elmo_SeqLabel(nn.Module):
    def __init__(self, data):
        super(Elmo_SeqLabel, self).__init__()
        self.use_crf = data.use_crf
        print("build elmo sequence labeling network...")
        print("use crf: ", self.use_crf)

        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss
        ## add two more label for downlayer lstm, use original label size for CRF
        label_size = data.label_alphabet_size
        data.label_alphabet_size += 2

        self.word_hidden = Elmo(data.elmo_options_file, data.elmo_weight_file, 1, requires_grad=data.elmo_tune, dropout=data.elmo_dropout)

        with open(data.elmo_options_file, 'r') as fin:
            self._options = json.load(fin)
        self.hidden2tag = nn.Linear(self._options['lstm']['projection_dim']*2, data.label_alphabet_size)

        if self.use_crf:
            self.crf = CRF(label_size, self.gpu)

        if self.gpu >= 0 and torch.cuda.is_available():
            self.word_hidden = self.word_hidden.cuda(self.gpu)
            self.hidden2tag = self.hidden2tag.cuda(self.gpu)


    def calculate_loss(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask):
        elmo_outputs = self.word_hidden(char_inputs)
        outs = elmo_outputs['elmo_representations'][0]
        # mask = elmo_outputs['mask']
        batch_size = char_inputs.size(0)
        seq_len = char_inputs.size(1)
        outs = self.hidden2tag(outs)
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


    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask):
        elmo_outputs = self.word_hidden(char_inputs)
        outs = elmo_outputs['elmo_representations'][0]
        # mask = elmo_outputs['mask']
        batch_size = char_inputs.size(0)
        seq_len = char_inputs.size(1)
        outs = self.hidden2tag(outs)
        if self.use_crf:
            scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        else:
            outs = outs.view(batch_size * seq_len, -1)
            _, tag_seq  = torch.max(outs, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            ## filter padded position with zero
            tag_seq = mask.long() * tag_seq
        return tag_seq

    def decode_nbest(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask, nbest):
        if not self.use_crf:
            print("Nbest output is currently supported only for CRF! Exit...")
            exit(0)
        elmo_outputs = self.word_hidden(char_inputs)
        outs = elmo_outputs['elmo_representations'][0]
        # mask = elmo_outputs['mask']
        outs = self.hidden2tag(outs)
        scores, tag_seq = self.crf._viterbi_decode_nbest(outs, mask, nbest)
        return scores, tag_seq