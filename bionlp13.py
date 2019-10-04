import codecs
import os

# replace '\t' with ' '
def tsv_to_txt(input_file, output_file):
    with codecs.open(input_file, 'r', 'UTF-8') as in_fp:
        with codecs.open(output_file, 'w', 'UTF-8') as out_fp:
            for in_line in in_fp:
                out_line = in_line.replace('\t', ' ')
                out_fp.write(out_line)


if __name__ == '__main__':
    # tsv_to_txt('/Users/feili/project/NCRFpp_0914/NCRFpp/BioNLP13PC-IOBES/train.tsv', '/Users/feili/project/NCRFpp_0914/NCRFpp/BioNLP13PC-IOBES/train.txt')
    # tsv_to_txt('/Users/feili/project/NCRFpp_0914/NCRFpp/BioNLP13PC-IOBES/devel.tsv', '/Users/feili/project/NCRFpp_0914/NCRFpp/BioNLP13PC-IOBES/devel.txt')
    # tsv_to_txt('/Users/feili/project/NCRFpp_0914/NCRFpp/BioNLP13PC-IOBES/test.tsv', '/Users/feili/project/NCRFpp_0914/NCRFpp/BioNLP13PC-IOBES/test.txt')


    tsv_to_txt('/Users/feili/project/NCRFpp_0914/NCRFpp/BioNLP13CG-IOBES/train.tsv', '/Users/feili/project/NCRFpp_0914/NCRFpp/BioNLP13CG-IOBES/train.txt')
    tsv_to_txt('/Users/feili/project/NCRFpp_0914/NCRFpp/BioNLP13CG-IOBES/devel.tsv', '/Users/feili/project/NCRFpp_0914/NCRFpp/BioNLP13CG-IOBES/devel.txt')
    tsv_to_txt('/Users/feili/project/NCRFpp_0914/NCRFpp/BioNLP13CG-IOBES/test.tsv', '/Users/feili/project/NCRFpp_0914/NCRFpp/BioNLP13CG-IOBES/test.txt')


    pass