from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_files
import numpy as np
import mglearn

reviews_train = load_files('data/train/')
text_train, y_train = reviews_train.data, reviews_train.target
vect = CountVectorizer(max_features=1000, max_df=.15)
X = vect.fit_transform(text_train)

from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=10, learning_method="batch",
                                max_iter=25, random_state=0)
document_topics = lda.fit_transform(X)
print('lda.components_.shape:{}\n'.format(lda.components_.shape))

sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
feature_names = np.array(vect.get_feature_names())
mglearn.tools.print_topics(topics=range(10), feature_names=feature_names,
                           sorting=sorting, topics_per_chunk=5, n_words=10)

