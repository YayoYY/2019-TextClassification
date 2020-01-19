import numpy as np
import pandas as pd
import fasttext
from sklearn.metrics import f1_score, classification_report

print('>>> reading...')
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

print('>>> change numpy format to fast text...')
strip = lambda x : x.replace("\'", '').replace('[', '').replace(']', '')

train['label'] = train['accusation'].apply(strip)
train['label'] = '__label__' + train['label'] + ' '
test['label'] = test['accusation'].apply(strip)
test['label'] = '__label__' + test['label']+' '
train['words'] = ' ' + train['words']
test['words'] = ' ' + test['words']

print('>>> saving data...')
train[['label', 'words']].to_csv('../data/fast_text_train.txt', index=None, header=None)
test[['label', 'words']].to_csv('../data/fast_text_test.txt', index=None, header=None)

print('>>> training...')
model = fasttext.train_supervised(input="fast_text_train.txt")

print('>>> evaluate...')
result = model.test('fast_text_test.txt')
print('result:', result)

print('>>> predicting...')
print('result:', model.predict(test.words[0], k=30)) # k控制输出top几个