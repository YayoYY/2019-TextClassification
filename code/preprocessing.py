import re
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

tokenize = lambda x: ' '.join([word for word in jieba.lcut(x, cut_all=False) if word not in stopwords])
delete_num_and_letter = lambda x: re.sub('[0-9A-za-z]', '', x)

print('>>> reading...')
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
with open('../data/stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = f.read().split('\n')
print('train size:', train.shape)
print('test size:', test.shape)
print('stopwords size:', len(stopwords))

print('>>> tokenizing...')
train['words'] = train['fact'].apply(tokenize)
test['words'] = test['fact'].apply(tokenize)
train['words'] = train['words'].apply(delete_num_and_letter)
test['words'] = test['words'].apply(delete_num_and_letter)

print('>>> saving...')
train.to_csv('../data/train.csv', index=None)
test.to_csv('../data/test.csv', index=None)

# >>> reading...
# Building prefix dict from the default dictionary ...
# Loading model from cache C:\Users\ADMINI~1\AppData\Local\Temp\5\jieba.cache
# train size: (32171, 35)
# test size: (3575, 35)
# stopwords size: 1893
# >>> tokenizing...
# Loading model cost 2.247 seconds.
# Prefix dict has been built succesfully.
# >>> saving...