from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

print('불용어 개수:', len(ENGLISH_STOP_WORDS))
print('매 10번째 불용어:\n', list(ENGLISH_STOP_WORDS)[::10])


reviews_train = load_files('data/train/')
text_train, y_train = reviews_train.data, reviews_train.target
vect = CountVectorizer(min_df=5, stop_words="english").fit(text_train)
X_train = vect.transform(text_train)
print("\n불용어가 제거된 X_train:\n", repr(X_train))


