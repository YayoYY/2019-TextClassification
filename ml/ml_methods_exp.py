import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from classifiers import *
from feature_engineering import *
import warnings
warnings.filterwarnings('ignore')

avgf1_scorer = lambda x,y: (f1_score(x, y, average='macro') + f1_score(x, y, average='micro')) / 2
labels = ['交通肇事', '信用卡诈骗', '危险驾驶', '容留他人吸毒', '寻衅滋事', '抢劫', '故意伤害', '盗窃', '诈骗', '走私、贩卖、运输、制造毒品']

def experiments(analyzer='word', ngram_range=(1,1)):
    print('==================================')
    print('analyzer=%s, ngram_range=%s' %(analyzer, str(ngram_range)))
    print('==================================')
    
    train, test, train_feature, test_feature, feature_names = vsm_tfidf(analyzer=analyzer, ngram_range=ngram_range)

    classifiers = [LR(), SVM(), KNN(), DT(), NB(), LGB()]
    for model in classifiers:
        print('-----------------------%s-----------------------' % model.name)
        print('>>> 模型训练...')
        model.train(train_feature, train[list(map(str, range(10)))].values)

        print('>>> 模型预测...')
        pred_y_prob = model.predict(test_feature)
        pred_y_prob_mod = (pred_y_prob > 0.22).astype(int)

        print('>>> 模型评估...')
        report = classification_report(test[list(map(str, range(10)))].values, pred_y_prob_mod, target_names=[x[:3]+'..\t' for x in labels])
        print(report)
        score = avgf1_scorer(test[list(map(str, range(10)))].values, pred_y_prob_mod)
        print('宏F1与微F1的均值：', score)

if __name__ == '__main__':
    experiments(analyzer='word', ngram_range=(1,1))
    experiments(analyzer='char', ngram_range=(2,2))
    experiments(analyzer='char', ngram_range=(2,3))