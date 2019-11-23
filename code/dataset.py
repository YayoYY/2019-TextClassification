import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

print('>>> reading...')
data = pd.read_csv('../data/data.csv')
print('data size:', data.shape)

print('>>> multilabel one-hot processing...')
data['accusation'] = data['accusation'].apply(lambda x: x.split(';'))
mlb = MultiLabelBinarizer()
label = pd.DataFrame(mlb.fit_transform(data.accusation), columns=list(range(30)))
data = pd.concat([data, label], axis=1)
print('labels:', mlb.classes_)
print('data shape:', data.shape)

print('>>> subset extracting...')
bigdataset, smalldataset= train_test_split(data, test_size=0.1, random_state=2019)
print('>>> train test splitting...')
train, test = train_test_split(smalldataset, test_size=0.1, random_state=2019)
print('train shape:', train.shape)
print('test shape:', test.shape)

print('>>> saving...')
train.to_csv('../data/train.csv', index=None)
test.to_csv('../data/test.csv', index=None)

# >>> reading...
# data size: (357454, 5)
# >>> multilabel one-hot processing...
# labels: ['交通肇事' '信用卡诈骗' '危险驾驶' '受贿' '合同诈骗' '妨害公务' '容留他人吸毒' '寻衅滋事' '开设赌场'
#  '引诱、容留、介绍卖淫' '抢劫' '抢夺' '掩饰、隐瞒犯罪所得、犯罪所得收益' '故意伤害' '故意杀人' '故意毁坏财物' '敲诈勒索'
#  '滥伐林木' '生产、销售假药' '盗窃' '组织、强迫、引诱、容留、介绍卖淫' '职务侵占' '诈骗' '贪污' '赌博'
#  '走私、贩卖、运输、制造毒品' '过失致人死亡' '非法拘禁' '非法持有、私藏枪支、弹药' '非法持有毒品']
# data shape: (357454, 35)
# >>> subset extracting...
# >>> train test splitting...
# train shape: (32171, 35)
# test shape: (3575, 35)
# >>> saving...