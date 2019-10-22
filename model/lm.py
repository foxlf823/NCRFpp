

from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wordsequence1 import WordSequence1

class LanguageModel(nn.Module):
    def __init__(self, data):
        super(LanguageModel, self).__init__()

        print("build language model...")
        print("use_char: ", data.use_char)
        if data.use_char:
            print("char feature extractor: ", data.char_feature_extractor)
        print("word feature extractor: ", data.word_feature_extractor)

        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss
        ## add two more label for downlayer lstm, use original label size for CRF
        label_size = data.label_alphabet_size
        data.label_alphabet_size += 2
        self.word_hidden = WordSequence1(data)

        self.bilstm_flag = data.HP_bilstm
        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            # lstm_hidden = data.HP_hidden_dim
            raise RuntimeError("not support single-lstm lm")
        # The linear layer that maps from hidden state space to tag space
        self.use_elmo = data.use_elmo
        if self.use_elmo:
            self.hidden2tag_forward = nn.Linear(lstm_hidden // 2, data.label_alphabet_size)
            self.hidden2tag_backward = nn.Linear(lstm_hidden // 2, data.label_alphabet_size)
        else:
            self.hidden2tag_forward = nn.Linear(lstm_hidden, data.label_alphabet_size)
            self.hidden2tag_backward = nn.Linear(lstm_hidden, data.label_alphabet_size)

        if self.gpu >= 0 and torch.cuda.is_available():
            self.hidden2tag_forward = self.hidden2tag_forward.cuda(self.gpu)
            self.hidden2tag_backward = self.hidden2tag_backward.cuda(self.gpu)
            

    def calculate_loss(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                       char_seq_recover, batch_label_forward, batch_label_backward, mask,
                       elmo_char_inputs):
        hidden_forward, hidden_backward = self.word_hidden(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                char_seq_recover, elmo_char_inputs)

        outs_forward = self.hidden2tag_forward(hidden_forward)
        outs_backward = self.hidden2tag_backward(hidden_backward)
        
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
        
        outs_forward = outs_forward.view(batch_size * seq_len, -1)
        score_forward = F.log_softmax(outs_forward, 1)
        total_loss_forward = loss_function(score_forward, batch_label_forward.view(batch_size * seq_len))
        _, tag_seq_forward = torch.max(score_forward, 1)
        tag_seq_forward = tag_seq_forward.view(batch_size, seq_len)
        
        if self.average_batch:
            total_loss_forward = total_loss_forward / batch_size
        
        outs_backward = outs_backward.view(batch_size * seq_len, -1)
        score_backward = F.log_softmax(outs_backward, 1)
        total_loss_backward = loss_function(score_backward, batch_label_backward.view(batch_size * seq_len))
        _, tag_seq_backward = torch.max(score_backward, 1)
        tag_seq_backward = tag_seq_backward.view(batch_size, seq_len)
        
        if self.average_batch:
            total_loss_backward = total_loss_backward / batch_size

        total_loss = (total_loss_forward + total_loss_backward)/2

        return total_loss, tag_seq_forward, tag_seq_backward

    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                char_seq_recover, mask,
                elmo_char_inputs):
        hidden_forward, hidden_backward = self.word_hidden(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                char_seq_recover, elmo_char_inputs)

        outs_forward = self.hidden2tag_forward(hidden_forward)
        outs_backward = self.hidden2tag_backward(hidden_backward)
        
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)

        outs_forward = outs_forward.view(batch_size * seq_len, -1)
        _, tag_seq_forward = torch.max(outs_forward, 1)
        tag_seq_forward = tag_seq_forward.view(batch_size, seq_len)
        ## filter padded position with zero
        tag_seq_forward = mask.long() * tag_seq_forward
        
        outs_backward = outs_backward.view(batch_size * seq_len, -1)
        _, tag_seq_backward = torch.max(outs_backward, 1)
        tag_seq_backward = tag_seq_backward.view(batch_size, seq_len)
        tag_seq_backward = mask.long() * tag_seq_backward
        
        return tag_seq_forward, tag_seq_backward