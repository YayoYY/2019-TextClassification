import pandas as pd
import jieba

remove_punctuation = lambda x: x.replace('。', '，').strip('，')
tokenize = lambda x: jieba.lcut(x)
remove_stopwords = lambda x: filter(lambda y: y not in stopwords, x)
delete_single = lambda x: filter(lambda y: len(y) > 1, x)
concat = lambda x: ' '.join(x)

print('>>> reading...')
data = pd.read_csv('../data/train.csv')
with open('../data/stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = f.read().split('\n')

print('>>> constructing corpus...')
sentences = []
for text in data.fact:
    text = remove_punctuation(text)
    text_sentences = text.split('，')
    text_sentences = [tokenize(sentence) for sentence in text_sentences]
    text_sentences = [remove_stopwords(sentence) for sentence in text_sentences]
    text_sentences = [delete_single(sentence) for sentence in text_sentences]
    text_sentences = [concat(sentence) for sentence in text_sentences]
    sentences += text_sentences

print('>>> saving...')
with open('../data/corpus.txt', 'w', encoding='utf-8') as f:
    f.writelines('\n'.join([sentence for sentence in set(sentences)]))
    
# >>> reading...
# Building prefix dict from the default dictionary ...
# >>> constructing corpus...
# Dumping model to file cache /var/folders/s3/pb8pb7h94b13hj_dmvrkj90m0000gn/T/jieba.cache
# Loading model cost 0.872 seconds.
# Prefix dict has been built succesfully.
# >>> saving...