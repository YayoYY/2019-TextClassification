import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif, chi2
import warnings
warnings.filterwarnings('ignore')

def vsm_tfidf(max_features=5000, max_df=0.8, analyzer='word', ngram_range=(1,1)):
    
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
    selector = SelectKBest(chi2, k=500)
    train_feature = selector.fit_transform(train_feature, train['accusation'])
    selected_features = selector.inverse_transform(train_feature)
    selected_columns = np.where(~(selector.inverse_transform(train_feature) == 0).all(axis=0))[0]
    test_feature = test_feature[:, selected_columns]
    print('选择后的特征：', [feature_names[i] for i in selected_columns])
    print('训练集维度：', train_feature.shape)
    print('测试集维度：', test_feature.shape)
    
    return train, test, train_feature, test_feature, feature_names