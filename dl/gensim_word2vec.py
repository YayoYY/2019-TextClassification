import collections
from gensim.models import word2vec
from gensim.models import KeyedVectors

corpus_path = '../data/corpus.txt'
model_path = '../data/word2vec_embedding.bin'

print('>>> word2vec training...')
sentences = word2vec.Text8Corpus(corpus_path)
# size词向量维度、window窗口大小、min_count最小词频数、iter随机梯度下降迭代最小次数   
model = word2vec.Word2Vec(sentences, size=100, window=8, min_count=3, iter=8)
model.wv.save_word2vec_format(model_path, binary=False)

print('>>> test word2vec...')
model = KeyedVectors.load_word2vec_format(model_path)
print('The top10 of 冰毒:\n {}'.format(model.most_similar("冰毒", topn=10)))


# >>> word2vec training...
# >>> test word2vec...
# The top10 of 冰毒:
#  [('氯胺', 0.6852064728736877), ('海洛因', 0.6748599410057068), ('麻果', 0.6653697490692139), 
#   ('氯胺酮', 0.6337580680847168), ('苯丙胺', 0.5960184335708618), ('物给', 0.5952740907669067), 
#   ('甲基苯丙胺', 0.5911368131637573), ('阿财', 0.5877951383590698), ('某乙帮', 0.585311770439148), 
#   ('麻古', 0.5789621472358704)]