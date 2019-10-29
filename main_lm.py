
import argparse
from utils.data import Data
import random
import torch
import numpy as np
import sys
import gc

seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

def data_initialization(data):
    data.initial_feature_alphabets()
    data.build_alphabet(data.train_dir)
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
    if sentence_classification:
        # print(overlaped)
        # print(overlaped*pred)
        right_token = np.sum(overlaped)
        total_token = overlaped.shape[0] ## =batch_size
    else:
        right_token = np.sum(overlaped * mask)
        total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token

def batchify_with_label(input_batch_list, input_text_batch_list, gpu, if_train=True, sentence_classification=False):
    if sentence_classification:
        # return batchify_sentence_classification_with_label(input_batch_list, gpu, if_train)
        raise RuntimeError("not support")
    else:
        return batchify_sequence_labeling_with_label(input_batch_list, input_text_batch_list, gpu, if_train)

from elmo.elmo import batch_to_ids

def batchify_sequence_labeling_with_label(input_batch_list, input_text_batch_list, gpu, if_train=True):

    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    features = [np.asarray(sent[1]) for sent in input_batch_list]
    feature_num = len(features[0][0])
    chars = [sent[2] for sent in input_batch_list]
    labels_forward = [sent[3] for sent in input_batch_list]
    labels_backward = [sent[4] for sent in input_batch_list]

    words_text = [sent[5] for sent in input_text_batch_list]
    elmo_char_seq_tensor = batch_to_ids(words_text)

    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).long()
    label_forward_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).long()
    label_backward_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad= if_train).long()
    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(torch.zeros((batch_size, max_seq_len),requires_grad =  if_train).long())
    mask = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).byte()
    for idx, (seq, label_forward, label_backward, seqlen) in enumerate(zip(words, labels_forward, labels_backward, word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_forward_seq_tensor[idx, :seqlen] = torch.LongTensor(label_forward)
        label_backward_seq_tensor[idx, :seqlen] = torch.LongTensor(label_backward)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
        for idy in range(feature_num):
            feature_seq_tensors[idy][idx,:seqlen] = torch.LongTensor(features[idx][:,idy])
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

    label_forward_seq_tensor = label_forward_seq_tensor[word_perm_idx]
    label_backward_seq_tensor = label_backward_seq_tensor[word_perm_idx]
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
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda(gpu)
        word_seq_lengths = word_seq_lengths.cuda(gpu)
        word_seq_recover = word_seq_recover.cuda(gpu)
        label_forward_seq_tensor = label_forward_seq_tensor.cuda(gpu)
        label_backward_seq_tensor = label_backward_seq_tensor.cuda(gpu)
        char_seq_tensor = char_seq_tensor.cuda(gpu)
        char_seq_recover = char_seq_recover.cuda(gpu)
        mask = mask.cuda(gpu)
        elmo_char_seq_tensor = elmo_char_seq_tensor.cuda(gpu)
    return word_seq_tensor,feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_forward_seq_tensor, label_backward_seq_tensor, mask, elmo_char_seq_tensor

from model.lm import LanguageModel
import torch.optim as optim
import time

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr/(1+decay_rate*epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def train(data):
    print("Training model...")
    data.show_data_summary()
    save_data_name = data.model_dir +".dset"
    data.save(save_data_name)

    model = LanguageModel(data)

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
            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label_forward, batch_label_backward, mask, batch_elmo_char = batchify_with_label(instance, instance_text, data.HP_gpu, True, data.sentence_classification)
            instance_count += 1
            loss, tag_seq_forward, tag_seq_backward = model.calculate_loss(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label_forward, batch_label_backward, mask, batch_elmo_char)
            right_forward, whole_forward = predict_check(tag_seq_forward, batch_label_forward, mask, data.sentence_classification)
            right_backward, whole_backward = predict_check(tag_seq_backward, batch_label_backward, mask, data.sentence_classification)

            right_token += (right_forward + right_backward)
            whole_token += (whole_forward + whole_backward)
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
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s, acc: %s/%s=%.4f"%(idx, epoch_cost, train_num/epoch_cost, total_loss, right_token, whole_token,(right_token+0.)/whole_token))
        print("totalloss:", total_loss)
        if total_loss > 1e8 or str(total_loss) == "nan":
            print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
            exit(1)
        # continue

        current_score = (right_token+0.)/whole_token

        if current_score > best_dev:

            print("Exceed previous best acc:", best_dev)
            # model_name = data.model_dir +'.'+ str(idx) + ".model"
            model_name = data.model_dir + ".lm.model"
            # print("Save current best model in file:", model_name)
            torch.save(model.state_dict(), model_name)
            best_dev = current_score

            bad_counter = 0
        else:
            bad_counter += 1

        gc.collect()

        if bad_counter >= data.patience:
            print('Early Stop!')
            break

if __name__ == '__main__':
    print('train a bilstm language model')
    parser = argparse.ArgumentParser()

    parser.add_argument('--config',  help='Configuration File', default='None')

    args = parser.parse_args()
    data = Data()

    if args.config == 'None':
        raise RuntimeError("must provide a config file")
    else:
        data.read_config(args.config)

    status = data.status.lower()
    print("Seed num:",seed_num)

    if status == 'train':
        print("MODEL: train")
        data_initialization(data)
        data.generate_instance('train')
        data.build_pretrain_emb()
        train(data)
    elif status == 'decode':
        print("Invalid argument!")
    else:
        print("Invalid argument!")