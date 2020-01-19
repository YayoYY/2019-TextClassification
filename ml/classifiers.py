import functools, time
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb

def logtime(func):
    '''计时装饰器'''
    
    @functools.wraps(func)
    def wrapper(*args, **kw):
        time1 = time.time()
        result = func(*args, **kw)
        time2 = time.time()
        print(func.__name__, 'time cost:', round(time2-time1, 4), 's')
        return result
    return wrapper

class Classifier():
    '''分类器'''
    
    def __init__(self):
        self.model = None
    
    @logtime
    def train(self, X, y):
        self.model.fit(X, y)
        
    @logtime
    def predict(self, X):
        return self.model.predict_proba(X)

class LR(Classifier):
    '''逻辑回归'''
    
    name = 'LR'
    def __init__(self, C=1.0, penalty='l2', solver='saga'):
        self.model = OneVsRestClassifier(LogisticRegression(solver=solver, 
                                                            class_weight='balanced',
                                                            C=C, 
                                                            penalty=penalty,
                                                            n_jobs=-1))
        
class SVM(Classifier):
    '''支持向量机'''
    
    name = 'SVM'
    def __init__(self, C=1.0, kernel='rbf', probability=True):
        self.model = OneVsRestClassifier(SVC(C=C,
                                             kernel=kernel, 
                                             probability=probability))
        
class KNN(Classifier):
    '''K近邻'''
    
    name = 'KNN'
    def __init__(self, n_neighbors=20, metric='euclidean', algorithm='ball_tree'):
        self.model = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=n_neighbors, 
                                                              metric=metric,
                                                              algorithm=algorithm,
                                                              n_jobs=-1)) 
class DT(Classifier):
    '''决策树'''
    
    name = 'DT'
    def __init__(self):
        self.model = OneVsRestClassifier(DecisionTreeClassifier(max_depth=3,
                                                                min_samples_leaf=5))
        
class NB(Classifier):
    '''朴素贝叶斯'''
    
    name = 'NB'
    def __init__(self):
        self.model = OneVsRestClassifier(GaussianNB())
        
class LGB(Classifier):
    '''LightGBM'''
    
    name = 'LGB'
    def __init__(self):
        self.model = OneVsRestClassifier(lgb.LGBMClassifier(num_leaves=2**5,
                                                            reg_alpha=0.25,
                                                            reg_lambda=0.25,
                                                            max_depth=-1, 
                                                            learning_rate=0.05,
                                                            min_child_sample=5,
                                                            n_estimators=200,
                                                            subsample=0.9,
                                                            colsample_bytree=0.7, 
                                                            objective='binary',
                                                            silent=200))