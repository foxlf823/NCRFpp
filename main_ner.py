# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:11:08
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-02-13 12:41:44

from __future__ import print_function
import time
import sys
import argparse
import random
import torch
import gc
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.metric import get_ner_fmeasure
from model.seqlabel1 import SeqLabel1
from model.sentclassifier import SentClassifier
from utils.data import Data

try:
    import cPickle as pickle
except ImportError:
    import pickle


seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def data_initialization(data):
    data.initial_feature_alphabets()
    data.build_alphabet(data.s_lm, build_label_alphabet=False)
    data.build_alphabet(data.t_lm, build_label_alphabet=False)
    if data.mode == 'ner':
        data.build_alphabet(data.s_ner_train, build_label_alphabet=False)
        data.build_alphabet(data.s_ner_eval, build_label_alphabet=False)
    elif data.mode == 'finetune':
        data.build_alphabet(data.s_ner_train, target=False)
        data.build_alphabet(data.s_ner_eval, target=False)
    else:
        raise RuntimeError("not support")
    data.build_alphabet(data.train_dir)
    data.build_alphabet(data.dev_dir)
    data.build_alphabet(data.test_dir)
    data.fix_alphabet()


def predict_check(pred_variable, gold_variable, mask_variable, sentence_classification=False):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)

    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover, sentence_classification=False):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)

    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert(len(pred)==len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len, nbest): pred tag result
            mask_variable (batch_size, sent_len): mask variable
            word_recover (batch_size)
        output:
            nbest_pred_label list: [batch_size, nbest, each_seq_len]
    """
    # exit(0)
    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = pred_variable.size(0)
    seq_len = pred_variable.size(1)
    nbest = pred_variable.size(2)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = []
        for idz in range(nbest):
            each_pred = [label_alphabet.get_instance(pred_tag[idx][idy][idz]) for idy in range(seq_len) if mask[idx][idy] != 0]
            pred.append(each_pred)
        pred_label.append(pred)
    return pred_label


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr/(1+decay_rate*epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer



def evaluate(data, model, name, label_alphabet, nbest=None):
    if name == "train":
        instances = data.train_Ids
        instances_text = data.train_texts
    elif name == "dev":
        instances = data.dev_Ids
        instances_text = data.dev_texts
    elif name == 'test':
        instances = data.test_Ids
        instances_text = data.test_texts
    elif name == 'raw':
        instances = data.raw_Ids
        instances_text = data.raw_texts
    elif name == 's_train':
        instances = data.s_train_Ids
        instances_text = data.s_train_texts
    elif name == 's_dev':
        instances = data.s_dev_Ids
        instances_text = data.s_dev_texts
    else:
        print("Error: wrong evaluate name,", name)
        exit(1)
    right_token = 0
    whole_token = 0
    nbest_pred_results = []
    pred_scores = []
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.eval()
    batch_size = data.HP_batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end =  train_num
        instance = instances[start:end]
        instance_text = instances_text[start:end]
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask, batch_elmo_char = batchify_with_label(instance, instance_text, data.HP_gpu, False, data.sentence_classification)
        if nbest and not data.sentence_classification:
            scores, nbest_tag_seq = model.decode_nbest(batch_word,batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask, nbest, batch_elmo_char)
            nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, label_alphabet, batch_wordrecover)
            nbest_pred_results += nbest_pred_result
            pred_scores += scores[batch_wordrecover].cpu().data.numpy().tolist()
            ## select the best sequence to evalurate
            tag_seq = nbest_tag_seq[:,:,0]
        else:
            tag_seq = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask, batch_elmo_char)
        # print("tag:",tag_seq)
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, label_alphabet, batch_wordrecover, data.sentence_classification)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    if nbest and not data.sentence_classification:
        return speed, acc, p, r, f, nbest_pred_results, pred_scores
    return speed, acc, p, r, f, pred_results, pred_scores


def batchify_with_label(input_batch_list, input_text_batch_list, gpu, if_train=True, sentence_classification=False):

    return batchify_sequence_labeling_with_label(input_batch_list, input_text_batch_list, gpu, if_train)

from elmo.elmo import batch_to_ids

def batchify_sequence_labeling_with_label(input_batch_list, input_text_batch_list, gpu, if_train=True):
    """
        input: list of words, chars and labels, various length. [[words, features, chars, labels],[words, features, chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            features: features ids for one sentence. (batch_size, sent_len, feature_num)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            labels: label ids for one sentence. (batch_size, sent_len)

        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            feature_seq_tensors: [(batch_size, max_sent_len),...] list of Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    features = [np.asarray(sent[1]) for sent in input_batch_list]
    feature_num = len(features[0][0])
    chars = [sent[2] for sent in input_batch_list]
    labels = [sent[3] for sent in input_batch_list]
    words_backward = [sent[4] for sent in input_batch_list]

    words_text = [sent[4] for sent in input_text_batch_list]
    elmo_char_seq_tensor = batch_to_ids(words_text)

    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).long()
    word_backward_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).long()
    label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).long()
    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(torch.zeros((batch_size, max_seq_len),requires_grad =  if_train).long())
    mask = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).byte()
    for idx, (seq, seq_backward, label, seqlen) in enumerate(zip(words, words_backward, labels, word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        word_backward_seq_tensor[idx, :seqlen] = torch.LongTensor(seq_backward)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
        for idy in range(feature_num):
            feature_seq_tensors[idy][idx,:seqlen] = torch.LongTensor(features[idx][:,idy])
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    word_backward_seq_tensor = word_backward_seq_tensor[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    elmo_char_seq_tensor = elmo_char_seq_tensor[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad =  if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu >= 0 and torch.cuda.is_available():
        word_seq_tensor = word_seq_tensor.cuda(gpu)
        word_backward_seq_tensor = word_backward_seq_tensor.cuda(gpu)
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda(gpu)
        word_seq_lengths = word_seq_lengths.cuda(gpu)
        word_seq_recover = word_seq_recover.cuda(gpu)
        label_seq_tensor = label_seq_tensor.cuda(gpu)
        char_seq_tensor = char_seq_tensor.cuda(gpu)
        char_seq_recover = char_seq_recover.cuda(gpu)
        mask = mask.cuda(gpu)
        elmo_char_seq_tensor = elmo_char_seq_tensor.cuda(gpu)
    return word_seq_tensor, word_backward_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask, elmo_char_seq_tensor

from model.lm import LanguageModel
import copy

def train(data):
    print("Training model...")
    data.show_data_summary()
    save_data_name = data.model_dir +".dset"
    data.save(save_data_name)

    if data.mode == 'ner':
        model = SeqLabel1(data)
    elif data.mode == 'finetune':
        s_model = SeqLabel1(data, target=False)
        model = SeqLabel1(data)
    elif data.mode == 'lm_ner':
        model = SeqLabel1(data)
        lm_state_dict = torch.load(data.lm_model_dir + ".lm.model")

        # delete hidden2tag layers
        del lm_state_dict['hidden2tag_forward.weight']
        del lm_state_dict['hidden2tag_forward.bias']
        del lm_state_dict['hidden2tag_backward.weight']
        del lm_state_dict['hidden2tag_backward.bias']

        model.load_state_dict(lm_state_dict, strict=False)

    if data.mode == 'finetune':
        if data.optimizer.lower() == "sgd":
            s_optimizer = optim.SGD(s_model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum, weight_decay=data.HP_l2)
        elif data.optimizer.lower() == "adam":
            s_optimizer = optim.Adam(s_model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
        elif data.optimizer.lower() == "rmsprop":
            s_optimizer = optim.RMSprop(s_model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
        else:
            print("Optimizer illegal: %s" % (data.optimizer))
            exit(1)

    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum,weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    else:
        print("Optimizer illegal: %s"%(data.optimizer))
        exit(1)

    def freeze_net(model):
        for p in model.word_hidden.wordrep.word_embedding.parameters():
            p.requires_grad = False
    if data.tune_wordemb == False:
        freeze_net(model)
        if data.mode == 'finetune':
            freeze_net(s_model)

    if data.mode == 'finetune':
        print("###source domain training begins###")
        best_dev = -10
        bad_counter = 0
        for idx in range(data.HP_iteration):
            epoch_start = time.time()
            if data.optimizer == "SGD":
                s_optimizer = lr_decay(s_optimizer, idx, data.HP_lr_decay, data.HP_lr)

            cc = list(zip(data.s_train_Ids, data.s_train_texts))
            random.shuffle(cc)
            data.s_train_Ids[:], data.s_train_texts[:] = zip(*cc)
            s_model.train()
            s_model.zero_grad()
            batch_size = data.HP_batch_size
            train_num = len(data.s_train_Ids)
            total_batch = train_num // batch_size + 1
            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > train_num:
                    end = train_num
                instance = data.s_train_Ids[start:end]
                instance_text = data.s_train_texts[start:end]
                if not instance:
                    continue
                batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask, batch_elmo_char = batchify_with_label(
                    instance, instance_text, data.HP_gpu, True, data.sentence_classification)
                loss, tag_seq = s_model.calculate_loss(batch_word, batch_features, batch_wordlen, batch_char,
                                                     batch_charlen, batch_charrecover, batch_label, mask,
                                                     batch_elmo_char)
                loss.backward()
                s_optimizer.step()
                s_model.zero_grad()

            speed, acc, p, r, f, _, _ = evaluate(data, s_model, "s_dev", data.s_label_alphabet)
            epoch_finish = time.time()
            epoch_cost = epoch_finish - epoch_start
            print("Epoch %s finished: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
            idx, epoch_cost, speed, acc, p, r, f))
            sys.stdout.flush()

            current_score = f
            if current_score > best_dev:
                print("Exceed previous best dev f score:", best_dev)
                best_dev = current_score

                bad_counter = 0
            else:
                bad_counter += 1

            gc.collect()

            if bad_counter >= data.patience:
                print('Early Stop!')
                break

        print("###source domain training ends###")

        s_model_state_dict = copy.deepcopy(s_model.state_dict())
        # delete hidden2tag layers
        del s_model_state_dict['hidden2tag.weight']
        del s_model_state_dict['hidden2tag.bias']
        if data.use_crf:
            del s_model_state_dict['crf.transitions']

        model.load_state_dict(s_model_state_dict, strict=False)

    best_dev = -10
    best_test = -10
    bad_counter = 0
    # data.HP_iteration = 1
    ## start training
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" %(idx,data.HP_iteration))
        if data.optimizer == "SGD":
            optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        sample_id = 0
        sample_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        cc = list(zip(data.train_Ids, data.train_texts))
        random.shuffle(cc)
        data.train_Ids[:], data.train_texts[:] = zip(*cc)
        print("Shuffle: first input word list:", data.train_Ids[0][0])
        ## set model in train model
        model.train()
        model.zero_grad()
        batch_size = data.HP_batch_size
        batch_id = 0
        train_num = len(data.train_Ids)
        total_batch = train_num//batch_size+1
        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end >train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            instance_text = data.train_texts[start:end]
            if not instance:
                continue
            batch_word, batch_word_backward, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask, batch_elmo_char = batchify_with_label(instance, instance_text, data.HP_gpu, True, data.sentence_classification)
            instance_count += 1
            loss, tag_seq = model.calculate_loss(batch_word, batch_word_backward, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask, batch_elmo_char)
            right, whole = predict_check(tag_seq, batch_label, mask, data.sentence_classification)
            right_token += right
            whole_token += whole
            # print("loss:",loss.item())
            sample_loss += loss.item()
            total_loss += loss.item()
            if end%500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token))
                if sample_loss > 1e8 or str(sample_loss) == "nan":
                    print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                    exit(1)
                sys.stdout.flush()
                sample_loss = 0
            loss.backward()
            optimizer.step()
            model.zero_grad()
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token))

        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s"%(idx, epoch_cost, train_num/epoch_cost, total_loss))
        if total_loss > 1e8 or str(total_loss) == "nan":
            print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
            exit(1)
        # continue
        speed, acc, p, r, f, _,_ = evaluate(data, model, "dev", data.label_alphabet)
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        if data.seg:
            current_score = f
            print("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(dev_cost, speed, acc, p, r, f))
        else:
            current_score = acc
            print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f"%(dev_cost, speed, acc))

        if current_score > best_dev:
            if data.seg:
                print("Exceed previous best dev f score:", best_dev)
            else:
                print("Exceed previous best dev acc score:", best_dev)
            # model_name = data.model_dir +'.'+ str(idx) + ".model"
            model_name = data.model_dir + ".dev.model"
            # print("Save current best model in file:", model_name)
            torch.save(model.state_dict(), model_name)
            best_dev = current_score

            bad_counter = 0
        else:
            bad_counter += 1
        # ## decode test
        speed, acc, p, r, f, _,_ = evaluate(data, model, "test", data.label_alphabet)
        test_finish = time.time()
        test_cost = test_finish - dev_finish
        if data.seg:
            current_score = f
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(test_cost, speed, acc, p, r, f))
        else:
            current_score = acc
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f"%(test_cost, speed, acc))

        if current_score > best_test:
            if data.seg:
                print("Exceed previous best test f score:", best_test)
            else:
                print("Exceed previous best test acc score:", best_test)

            model_name = data.model_dir + ".test.model"
            torch.save(model.state_dict(), model_name)
            best_test = current_score

        gc.collect()

        if bad_counter >= data.patience:
            print('Early Stop!')
            break




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with NCRF++')
    # parser.add_argument('--status', choices=['train', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--config',  help='Configuration File', default='None')
    parser.add_argument('--wordemb',  help='Embedding for words', default='None')
    parser.add_argument('--charemb',  help='Embedding for chars', default='None')
    parser.add_argument('--status', choices=['train', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--savemodel', default="data/model/saved_model.lstmcrf.")
    parser.add_argument('--savedset', help='Dir of saved data setting')
    parser.add_argument('--train', default="data/conll03/train.bmes") 
    parser.add_argument('--dev', default="data/conll03/dev.bmes" )  
    parser.add_argument('--test', default="data/conll03/test.bmes") 
    parser.add_argument('--seg', default="True") 
    parser.add_argument('--raw') 
    parser.add_argument('--loadmodel')
    parser.add_argument('--output')

    args = parser.parse_args()
    data = Data()
    # data.HP_gpu = torch.cuda.is_available()
    if args.config == 'None':
        data.train_dir = args.train 
        data.dev_dir = args.dev 
        data.test_dir = args.test
        data.model_dir = args.savemodel
        data.dset_dir = args.savedset
        print("Save dset directory:",data.dset_dir)
        save_model_dir = args.savemodel
        data.word_emb_dir = args.wordemb
        data.char_emb_dir = args.charemb
        if args.seg.lower() == 'true':
            data.seg = True
        else:
            data.seg = False
        print("Seed num:",seed_num)
    else:
        data.read_config(args.config)
    # data.show_data_summary()
    status = data.status.lower()
    print("Seed num:",seed_num)

    if status == 'train':
        data_initialization(data)
        if data.mode == 'finetune':
            data.generate_instance('s_train')
            data.generate_instance('s_dev')
        data.generate_instance('train')
        data.generate_instance('dev')
        data.generate_instance('test')
        data.build_pretrain_emb()
        train(data)
    else:
        print("Invalid argument! Please use valid arguments! (train/test/decode)")

