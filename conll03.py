import os

# after running this function, we need to manually remove the blank line in the head.
def transfer_neuroner_into_ncrfpp(input_dir, output_dir, tag='BIO'):
    fin_train = open(os.path.join(input_dir, 'train.txt'), 'r')
    fin_dev = open(os.path.join(input_dir, 'valid.txt'), 'r')
    fin_test = open(os.path.join(input_dir, 'test.txt'), 'r')

    fout_train = open(os.path.join(output_dir, 'train.txt'), 'w')
    fout_dev = open(os.path.join(output_dir, 'valid.txt'), 'w')
    fout_test = open(os.path.join(output_dir, 'test.txt'), 'w')

    fin = [fin_train, fin_dev, fin_test]
    fout = [fout_train, fout_dev, fout_test]

    for fin_, fout_ in zip(fin, fout):
        out_lines = []
        last_label = ''
        for line in fin_:
            line = line.strip()
            if line == '':
                if tag == 'BIOES':

                    if len(out_lines) > 0 and last_label == 'B':
                        last_out_line = out_lines[-1]
                        position = last_out_line.rfind('B-')
                        last_out_line = last_out_line[:position] + 'S' + last_out_line[position + 1:]
                        out_lines[-1] = last_out_line
                    elif len(out_lines) > 0 and last_label == 'I':
                        last_out_line = out_lines[-1]
                        position = last_out_line.rfind('I-')
                        last_out_line = last_out_line[:position] + 'E' + last_out_line[position + 1:]
                        out_lines[-1] = last_out_line

                out_lines.append('\n')
                last_label = ''
                continue
            if line.find('-DOCSTART-') != -1:
                continue

            columns = line.split()
            out_line = ''
            for idx, column in enumerate(columns):
                if idx == 0:
                    out_line += column+' '
                elif idx == 1:
                    #out_line += '[POS]'+column+" "
                    pass
                elif idx == 2:
                    pass
                else:
                    if tag == 'BIOES':
                        if column[0]=='B':
                            out_line += column+'\n'
                        elif column[0]=='I':
                            out_line += column+'\n'
                        elif column[0] == 'O':
                            if len(out_lines) > 0 and last_label=='B':
                                last_out_line = out_lines[-1]
                                position = last_out_line.rfind('B-')
                                last_out_line = last_out_line[:position]+'S'+last_out_line[position+1:]
                                out_lines[-1] = last_out_line
                            elif len(out_lines) > 0 and last_label=='I':
                                last_out_line = out_lines[-1]
                                position = last_out_line.rfind('I-')
                                last_out_line = last_out_line[:position] + 'E' + last_out_line[position + 1:]
                                out_lines[-1] = last_out_line
                            out_line += column+'\n'

                    else:
                        out_line += column+'\n'
            out_lines.append(out_line)
            last_label = column[0]

        for out_line in out_lines:
            fout_.write(out_line)

    fin_train.close()
    fin_dev.close()
    fin_test.close()

    fout_train.close()
    fout_dev.close()
    fout_test.close()

if __name__ == '__main__':
    transfer_neuroner_into_ncrfpp('/Users/feili/project/NeuroNER/data/conll2003/en',
                                  '/Users/feili/project/NCRFpp_0914/NCRFpp/conll03', tag='BIOES')

    pass