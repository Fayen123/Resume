import pkuseg
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

seg = pkuseg.pkuseg()


def is_chinese(uchar):
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False


def format_str(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str = content_str + i
    return content_str


filenames = ['a.txt', 'b.txt', 'c.txt']

if __name__ == '__main__':
    corpus = []
    for name in filenames:
        with open(name, 'r') as f:
            str = f.read()
            str = format_str(str)
            str = seg.cut(str)
            corpus.append(" ".join(str))
    print(corpus)
    vectorizer=CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    word = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    for (name, w) in zip(filenames, weight):
        print(name, ': ')
        loc = np.argsort(-w)
        for i in range(5):
            print(i + 1, word[loc[i]], w[loc[i]])
