from sklearn.feature_extraction.text import CountVectorizer

bards_words =["The fool doth think he is wise,",
              "but the wise man knows himself to be a fool"]

cv = CountVectorizer(ngram_range=(2, 2)).fit(bards_words)
print("어휘 사전 크기:", len(cv.vocabulary_))
print("어휘 사전:\n", cv.get_feature_names())

print('변환된 데이터:\n{}'.format(cv.transform(bards_words).toarray()))
