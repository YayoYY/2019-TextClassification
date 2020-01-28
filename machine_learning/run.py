import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
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

    print('>>> tf-idf 计算特征权重...')
    if analyzer == 'char':
        train.words = train.words.apply(lambda x: x.replace(' ', ''))
        test.words = test.words.apply(lambda x: x.replace(' ', ''))
    tfidfer = TfidfVectorizer(max_features=max_features, max_df=max_df, analyzer=analyzer, ngram_range=ngram_range)
    tfidfer.fit(train.words)
    feature_names = tfidfer.get_feature_names()
    tfidf = tfidfer.transform(train.words)
    train_feature = tfidf.toarray()
    tfidf = tfidfer.transform(test.words)
    test_feature = tfidf.toarray()
    print('特征：', feature_names[:20])
    print('训练集维度：', train_feature.shape)
    print('测试集维度：', test_feature.shape)
    
    print('>>> 特征选择...')
    selector = SelectKBest(chi2, k=num_features)
    train_feature = selector.fit_transform(train_feature, train['accusation'])
    selected_features = selector.inverse_transform(train_feature)
    selected_columns = np.where(~(selector.inverse_transform(train_feature) == 0).all(axis=0))[0]
    test_feature = test_feature[:, selected_columns]
    print('选择后的特征：', [feature_names[i] for i in selected_columns])
    print('训练集维度：', train_feature.shape)
    print('测试集维度：', test_feature.shape)
    
    return train, test, train_feature, test_feature, feature_names

def experiments(analyzer='word', ngram_range=(1,1)):
    '''实验'''

    print('==================================')
    print('analyzer=%s, ngram_range=%s' %(analyzer, str(ngram_range)))
    print('==================================')
    
    train, test, train_feature, test_feature, feature_names = get_feature(analyzer=analyzer, ngram_range=ngram_range, num_features=500)

    classifiers = [LR(), SVM(), LGB()]
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
    experiments(analyzer='word', ngram_range=(1,1)) # 分词
    experiments(analyzer='char', ngram_range=(2,2)) # 二字串
    experiments(analyzer='char', ngram_range=(2,3)) # 三字串