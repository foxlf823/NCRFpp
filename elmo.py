

def use_elmo_as_pytorch_module():
    # from allennlp.modules.elmo import Elmo, batch_to_ids
    from elmo.elmo import Elmo, batch_to_ids

    options_file = "elmo_small/elmo_2x1024_128_2048cnn_1xhighway_options.json"
    weight_file = "elmo_small/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"

    # Compute two different representation for each token.
    # Each representation is a linear weighted combination for the
    # 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
    elmo = Elmo(options_file, weight_file, 1, dropout=0)

    # use batch_to_ids to convert sentences to character ids
    sentences = [['First', 'sentence', '.'], ['Another', '.']]
    character_ids = batch_to_ids(sentences)

    embeddings = elmo(character_ids)

    # embeddings['elmo_representations'] is length two list of tensors.
    # Each element contains one layer of ELMo representations with shape
    # (2, 3, 1024).
    #   2    - the batch size
    #   3    - the sequence length of the batch
    #   1024 - the length of each ELMo vector
    pass

import codecs
def make_senna_embedding(word_file, value_file):
    out_fp = codecs.open('senna_emb_50d.txt', 'w', 'UTF-8')

    with codecs.open(word_file, 'r', 'UTF-8') as w_fp:
        with codecs.open(value_file, 'r', 'UTF-8') as v_fp:
            word_lines = w_fp.readlines()
            value_lines = v_fp.readlines()
            for word, value in zip(word_lines, value_lines):
                word = word.strip()
                value = value.strip()
                out_fp.write(word+" "+value+"\n")

    out_fp.close()

if __name__ == '__main__':
    # use_elmo_as_pytorch_module()

    make_senna_embedding('/Users/feili/Downloads/senna/hash/words.lst', '/Users/feili/Downloads/senna/embeddings/embeddings.txt')

    pass