bards_words =["The fool doth think he is wise,",
              "but the wise man knows himself to be a fool"]

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(bards_words)

print("어휘 사전의 크기:", len(vect.vocabulary_))
print("어휘 사전의 내용:\n", vect.vocabulary_)

bag_of_words = vect.transform(bards_words)
print("BOW:", repr(bag_of_words))

print("BOW의 밀집 표현:\n", bag_of_words.toarray())
'''
[[9, 3, 2, 10, 4, 6, 12]
 [1, 9, 12, 8, 7, 5, 11, 0, 3]]
'''
