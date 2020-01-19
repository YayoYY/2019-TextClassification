import jieba
import re
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')


tokenize = lambda x: jieba.lcut(x)
remove_stopwords = lambda x: filter(lambda y: y not in stopwords, x)
delete_single = lambda x: filter(lambda y: len(y) > 1, x)
concat = lambda x: ' '.join(x)

print('>>> 读取数据...')
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
with open('../data/stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = f.read().split('\n')
print('训练集维度：', train.shape)
print('测试集维度：', test.shape)
print('停用词维度：', len(stopwords))

print('>>> One Hot 编码...')
ohe = OneHotEncoder()
ohe.fit(train.accusation.values.reshape(-1,1))
label = pd.DataFrame(ohe.transform(train.accusation.values.reshape(-1,1)).toarray(), columns=list(range(10)))
train = pd.concat([train, label], axis=1)
label = pd.DataFrame(ohe.transform(test.accusation.values.reshape(-1,1)).toarray(), columns=list(range(10)))
test = pd.concat([test, label], axis=1)
print('训练集维度：', train.shape)
print('测试集维度：', test.shape)
print('类别：', list(ohe.categories_[0]))

print('>>> tokenizing and cleaning...')
train['words'] = train['fact'].apply(tokenize)
test['words'] = test['fact'].apply(tokenize)
train['words'] = train['words'].apply(remove_stopwords)
test['words'] = test['words'].apply(remove_stopwords)
train['words'] = train['words'].apply(delete_single)
test['words'] = test['words'].apply(delete_single)
train['words'] = train['words'].apply(concat)
test['words'] = test['words'].apply(concat)

print('>>> 保存...')
train.to_csv('../data/train.csv', index=None)
test.to_csv('../data/test.csv', index=None)

# >>> 读取数据...
# Building prefix dict from the default dictionary ...
# Loading model from cache /var/folders/s3/pb8pb7h94b13hj_dmvrkj90m0000gn/T/jieba.cache
# 训练集维度： (30000, 2)
# 测试集维度： (6000, 2)
# 停用词维度： 1893
# >>> One Hot 编码...
# 训练集维度： (30000, 12)
# 测试集维度： (6000, 12)
# 类别： ['交通肇事', '信用卡诈骗', '危险驾驶', '容留他人吸毒', '寻衅滋事', '抢劫', '故意伤害', '盗窃', '诈骗', '走私、贩卖、运输、制造毒品']
# >>> tokenizing and cleaning...
# Loading model cost 0.836 seconds.
# Prefix dict has been built succesfully.
# >>> 保存...