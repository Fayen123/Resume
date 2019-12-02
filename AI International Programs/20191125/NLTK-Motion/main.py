import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

if __name__ == '__main__':
    positive_fields = movie_reviews.fileids('pos')
    negative_fields = movie_reviews.fileids('neg')


def extract_features(word_list):
    return dict([(word, True) for word in word_list])


features_positive = [(extract_features(movie_reviews.words(fileids=[f])), 'Positive') for f in positive_fields]
features_negative = [(extract_features(movie_reviews.words(fileids=[f])), 'Negative') for f in negative_fields]
threshold_factor = 0.8
threshold_positive = int(threshold_factor * len(features_positive))
threshold_negative = int(threshold_factor * len(features_negative))
features_train = features_positive[:threshold_positive] + features_negative[:threshold_negative]
features_test = features_positive[threshold_positive:] + features_negative[threshold_negative:]

# Train a Native Bayes classifier
classifier = NaiveBayesClassifier.train(features_train)
print('\nAccuracy of the classifier:', nltk.classify.util.accuracy(classifier, features_test))
print('\nTop 10 most informative words:')
for item in classifier.most_informative_features()[:10]:
    print(item[0])

input_reviews = ["It is an amazing movie",
                 "This is a dull movie. I would never recommend it toanyone.",
                 "A complete and utter destruction of one of the most iconic superheroes. 0.1 effort and thought put "
                 "into the storyline. A coming of age awkward teenage movie with a 'spiderman' stamp put on it. Bad "
                 "jokes aimed at teenagers (at best). A complete caricature of a villain, a complete caricature of a "
                 "Spiderman. Just please stop making this garbage Put some god damn effort! A total waste of time",
                 "Just staving off some negative reviews. Fits well into the Marvel movies to date and is an "
                 "excellent follow up to Avengers: Endgame. "
                 ]
print('\nPredictions:')
for review in input_reviews:
    print('\nReview:', review)
    probdist = classifier.prob_classify(extract_features(review.split()))
    pred_sentiment = probdist.max()
    print('Predicted sentiment:', pred_sentiment)
    print('Probability:', round(probdist.prob(pred_sentiment), 2))


