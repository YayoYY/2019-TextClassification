import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

print('>>> 读取数据...')
data = pd.read_csv('../data/data.csv')
print('数据维度：', data.shape)

print('>>> 数据构建...')
data['accusation'] = data['accusation'].apply(lambda x: x.split(';')[0])
top10 = data.accusation.value_counts().sort_values(ascending=False).index.to_list()[:10]
data = data.loc[data.accusation.isin(top10)]

print('>>> 数据探索性分析...')
labels = []
for acc in data.accusation:
    labels.extend(acc.split(';'))
counter = Counter(labels)
print('类别：', counter.keys())
print('各类别样本数：\n', dict(counter))
print('最少类别样本数：', min(counter.values()))

print('>>> 数据集划分...')
columns = ['accusation', 'fact']
data = data[columns]
_ = [data.loc[data.accusation == acc][:3900] for acc in counter.keys()]
data = pd.concat(_, axis=0)
train, test = train_test_split(data, test_size=2/13, stratify=data.accusation)
train, dev = train_test_split(train, test_size=1/11, stratify=train.accusation)
print('训练集维度：', train.shape)
print('验证集维度：', dev.shape)
print('测试集维度：', test.shape)
train.to_csv('../data/train.csv', index=None)
test.to_csv('../data/test.csv', index=None)
dev.to_csv('../data/dev.csv', index=None)
# >>> 读取数据...
# 数据维度： (357454, 5)
# >>> 数据构建...
# >>> 数据探索性分析...
# 各类别样本数：
#  {'危险驾驶': 76527, '交通肇事': 36709, '盗窃': 80206, '信用卡诈骗': 5153, '容留他人吸毒': 13720, '故意伤害': 42809, '抢劫': 6128, '寻衅滋事': 7791, '走私、贩卖、运输、制造毒品': 27157, '诈骗': 11193}
# 最少类别样本数： 5153
# >>> 数据集划分...
# 训练集维度： (30000, 2)
# 验证集维度： (3000, 2)
# 测试集维度： (6000, 2)