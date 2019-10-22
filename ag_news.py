
import codecs
import re

corpus_path = './newsSpace'

# def clean(input_path, output_path):
#     text = b''
#     out_fp = open(output_path, 'wb')
#     with open(input_path, 'rb') as fp:
#         while True:
#             byte = fp.read(1)
#             if byte == '':
#                 break
#             elif byte == b'\x0D':
#                 text += b' '
#             else:
#                 text += byte
#
#     out_fp.write(text)
#     out_fp.close()

# def clean(input_path, output_path):
#     out_fp = codecs.open(output_path, 'w', 'utf-8')
#     in_fp = codecs.open(input_path, 'r', 'utf-8')
#     while True:
#         try:
#             text = in_fp.readlines()
#             if text == '':
#                 break
#             text = text.replace('\r', ' ')
#             out_fp.write(text)
#         except Exception as e:
#             pass
#
#     in_fp.close()
#     out_fp.close()

# source_filter = {'Yahoo World', 'Yahoo Business', 'Yahoo Sports', 'Yahoo News',
#                  'Yahoo U.S.', 'Yahoo Politics', 'Yahoo Tech', 'Yahoo Science', 'Yahoo Health', 'Yahoo Europe'}
source_filter = {'Reuters World', 'Reuters Top News', 'Reuters Business', 'Reuters AlertNet', 'Reuters U.S.', 'Reuters',
                 'Reuters Life and Leisure', 'Reuters Health', 'Reuters Tech', 'Reuters Entertainment', 'Reuters Sports'}

def read(input_path):
    all_news = []
    fp = open(input_path, 'r', encoding='iso-8859-1', newline='\n')
    one_news = ''
    while True:
        line1 = fp.readline()
        # if line1.find('http://sify.com/movies/fullstory.php?id=13547695') != -1:
        #     pass
        # elif line1.find('Despite age-old Olympic truce known as the ekecheiria') != -1:
        #     print('1')
        # if line1.find('<p><a href="http://us.rd.yahoo.com/dailynews/rss/elections/*http://news.yahoo.com/s/ap/20071215/ap_on_el_pr/obama_indicted_donor">') != -1:
        #     print("1")
        if line1 == '':
            break

        line = line1.rstrip()
        if line[-1] == '\\':
            line = line[:-1]+" "
            one_news += line
        elif line[-2:] == '\\N':
            one_news += line

            # replace \ to whitespace
            one_news = one_news.replace('\\', '')
            # split by \t+
            tabs = one_news.split('\t')
            # tabs = re.split(r"\t+", one_news)

            if tabs[0] not in source_filter:
                one_news = ''
                continue
            if len(tabs) > 11:
                one_news = ''
                continue


            if len(tabs) == 9:
                source = tabs[0].strip()
                url = tabs[1].strip()
                title = tabs[2].strip()
                image = tabs[3].strip()
                category = tabs[4].strip()
                description = tabs[5].strip()
                rank = tabs[6].strip()
                pubdate = tabs[7].strip()
                video = tabs[8].strip()
            # elif len(tabs) == 11:
            #     source = tabs[0].strip()
            #     url = tabs[1].strip()
            #     title = tabs[4].strip()
            #     image = tabs[5].strip()
            #     category = tabs[6].strip()
            #     description = tabs[7].strip()
            #     rank = tabs[8].strip()
            #     pubdate = tabs[9].strip()
            #     video = tabs[10].strip()
            elif len(tabs) == 10:
                source = tabs[0].strip()
                url = tabs[1].strip()
                title = tabs[2].strip()
                image = tabs[3].strip()
                category = tabs[4].strip()
                description = tabs[5].strip()+" "+tabs[6].strip()
                rank = tabs[7].strip()
                pubdate = tabs[8].strip()
                video = tabs[9].strip()
            else:
                raise RuntimeError("format error")

            all_news.append({'source':source, 'url':url, 'title':title, 'image':image, 'category':category, 'description':description,
                             'rank':rank, 'pubdate':pubdate, 'video':video})
            one_news = ''
        else:
            raise RuntimeError("format error")

    fp.close()
    return all_news

from my_utils import my_tokenize
import nltk

if __name__ == '__main__':
    # clean('./newsSpace', './newsSpace_clean')
    all_news = read('./newsSpace')

    # do statistics
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    source = set()
    type = set()
    typed_sentences = {}
    for news in all_news:
        source.add(news['source'])
        type.add(news['category'])
        if news['category'] in typed_sentences:
            sentences = typed_sentences[news['category']]
        else:
            sentences = []
            typed_sentences[news['category']] = sentences
        sentences.append(news['title'])

        all_sents_inds = []
        generator = sent_tokenizer.span_tokenize(news['description'])
        for t in generator:
            all_sents_inds.append(t)

        for ind in range(len(all_sents_inds)):
            t_start = all_sents_inds[ind][0]
            t_end = all_sents_inds[ind][1]
            sent_text = news['description'][t_start:t_end]
            sentences.append(sent_text)


    print("source: {}".format(source))
    print("type: {}".format(type))
    print("news number: {}".format(len(all_news)))
    for type, sentences in typed_sentences.items():
        print("type: {}, sent num: {}".format(type, len(sentences)))


