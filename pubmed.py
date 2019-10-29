import  os
import nltk
import codecs
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
from my_utils import my_tokenize

def fun1(dir):

    sent_num = 0

    for input_file_name in os.listdir(dir):
        if input_file_name.find(".txt") == -1:
            continue

        with codecs.open(os.path.join(dir, input_file_name), 'r', 'utf-8') as f:
            text = ''
            for line in f:
                line = line.strip()
                if line.find("|t|") != -1:
                    p = line.find("|t|")
                    text += line[p+len("|t|"):]+ " "
                elif line.find("|a|") != -1:
                    p = line.find("|a|")
                    text += line[p + len("|a|"):]

            all_sents_inds = []
            generator = sent_tokenizer.span_tokenize(text)
            for t in generator:
                all_sents_inds.append(t)

            sent_num += len(all_sents_inds)

    print("sent number: {}".format(sent_num))

# transfer pubmed raw text into conll format
# one line one word, sentence is split by \n
def pubmed_to_conll(dir, out_file):
    sent_num = 0
    out_f = codecs.open(out_file, 'w', 'utf-8')

    for input_file_name in os.listdir(dir):
        if input_file_name.find(".txt") == -1:
            continue

        with codecs.open(os.path.join(dir, input_file_name), 'r', 'utf-8') as f:
            text = ''
            for line in f:
                line = line.strip()
                if line.find("|t|") != -1:
                    p = line.find("|t|")
                    text += line[p+len("|t|"):]+ " "
                elif line.find("|a|") != -1:
                    p = line.find("|a|")
                    text += line[p + len("|a|"):]

            all_sents_inds = []
            generator = sent_tokenizer.span_tokenize(text)
            for t in generator:
                all_sents_inds.append(t)

            for ind in range(len(all_sents_inds)):
                t_start = all_sents_inds[ind][0]
                t_end = all_sents_inds[ind][1]
                sent_text = text[t_start:t_end]

                tokens = my_tokenize(sent_text)
                for token in tokens:
                    out_f.write(token+"\n")

                out_f.write("\n")
                sent_num += 1

    out_f.close()
    print("write {} into {}".format(sent_num, out_file))



if __name__ == '__main__':


    # fun1('/Users/feili/resource/pubmed_cancer_gene_pathway')
    # pubmed_to_conll('/Users/feili/resource/pubmed_cancer_gene_pathway', 'lm_data/lm_bio.txt')

    pass