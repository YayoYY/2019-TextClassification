import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from ml_model import *
import warnings
warnings.filterwarnings('ignore')

avgf1_scorer = lambda x,y: (f1_score(x, y, average='macro') + f1_score(x, y, average='micro')) / 2
labels = ['交通肇事', '信用卡诈骗', '危险驾驶', '容留他人吸毒', '寻衅滋事', '抢劫', '故意伤害', '盗窃', '诈骗', '走私、贩卖、运输、制造毒品']

def get_feature(max_features=5000, max_df=0.8, analyzer='word', ngram_range=(1,1), num_features=500):
    '''提取tf_idf特征'''

    print('>>> 读取数据...')
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    print('训练集维度：', train.shape)
    print('测试集维度：', test.shape)

    print('>>> 建立词汇表，进行CountVectorize...')
    if analyzer == 'char':
        train.words = train.words.apply(lambda x: x.replace(' ', ''))
        test.words = test.words.apply(lambda x: x.replace(' ', ''))
        cver = CountVectorizer(analyzer='char',
                               max_df=0.3,
                               min_df=0.001,
                               ngram_range=ngram_range)
    else:
        cver = CountVectorizer(max_df=0.3,
                               min_df=0.001,
                               ngram_range=ngram_range) # 先用文档频率过滤一下
    count_array = cver.fit_transform(train['words'])
    feature_names = cver.get_feature_names()
    n_vocabs = len(feature_names)
    print("词汇表数量：", n_vocabs)

    print('>>> chi2特征选择')
    selector = SelectKBest(chi2, k=500)
    train_feature = selector.fit_transform(count_array, train['accusation'])
    select_feature_names_bool = selector.get_support().tolist()
    select_feature_names = [feature_names[i] for i, item in enumerate(select_feature_names_bool) if item == 1]
    print('特征：', select_feature_names[:20])

    print('>>> tf-idf 计算特征权重...')
    tfidfer = TfidfVectorizer(max_features=max_features,
                              max_df=max_df,
                              analyzer=analyzer,
                              ngram_range=ngram_range,
                              vocabulary=select_feature_names)
    tfidfer.fit(train.words)
    feature_names = tfidfer.get_feature_names()
    tfidf = tfidfer.transform(train.words)
    train_feature = tfidf.toarray()
    tfidf = tfidfer.transform(test.words)
    test_feature = tfidf.toarray()
    print('训练集维度：', train_feature.shape)
    print('测试集维度：', test_feature.shape)
    
    return train, test, train_feature, test_feature, feature_names

def experiments(analyzer='word', ngram_range=(1,1)):
    '''实验'''

    print('==================================')
    print('analyzer=%s, ngram_range=%s' %(analyzer, str(ngram_range)))
    print('==================================')
    
    train, test, train_feature, test_feature, feature_names = get_feature(analyzer=analyzer, ngram_range=ngram_range, num_features=500)

    # classifiers = [LR(), SVM(), LGB()]
    classifiers = [LGB()]
    for model in classifiers:
        print('-----------------------%s-----------------------' % model.name)
        print('>>> 模型训练...')
        model.train(train_feature, train[list(map(str, range(10)))].values)

        print('>>> 模型预测...')
        pred_y_prob = model.predict(test_feature)
        pred_y_prob_mod = (pred_y_prob > 0.3).astype(int)

        print('>>> 模型评估...')
        report = classification_report(test[list(map(str, range(10)))].values, pred_y_prob_mod, target_names=[x[:3]+'..\t' for x in labels])
        print(report)
        score = avgf1_scorer(test[list(map(str, range(10)))].values, pred_y_prob_mod)
        print('宏F1与微F1的均值：', score)

if __name__ == '__main__':
    # experiments(analyzer='word', ngram_range=(1,1)) # 分词
    experiments(analyzer='char', ngram_range=(2,2)) # 二字串
    # experiments(analyzer='char', ngram_range=(2,3)) # 三字串