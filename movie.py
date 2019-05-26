import nltk
import sklearn
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import sys

def modelNaiveBayesAndSupportVector(trainDir, testDir, outNB, outLSV):
    #load in files and split 80/20 into testing and training sets
    reviews = load_files(trainDir, shuffle=True)
    X_train, X_test, y_train, y_test = train_test_split(reviews.data, reviews.target, test_size = 0.2, random_state=0)

    #Create Term-Feature/Inverse Document Feature Vectorizer
    vectorizer = TfidfVectorizer(min_df=3, max_features=4000, stop_words='english', tokenizer=nltk.word_tokenize)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    #Train/Test a Naive Bayes Classifier on the dataset
    NBClassifier = MultinomialNB()
    NBClassifier.fit(X_train_tfidf, y_train)
    NB_y_pred = NBClassifier.predict(X_test_tfidf)
    outputNB = open(outNB, 'w')
    outputNB.write('accuracy of model: ')
    outputNB.write('%.2f%%\n' % (accuracy_score(y_test, NB_y_pred)*100))
    for review, cls in zip(X_test, NB_y_pred):
        outputNB.write('%s | %r\n' % (reviews.target_names[cls], review))

    #Train/Test a Linear Support Vector Classifier on the dataset
    LSVClassifier = LinearSVC()
    LSVClassifier.fit(X_train_tfidf, y_train)
    LSV_y_pred = LSVClassifier.predict(X_test_tfidf)
    outputLSV = open(outLSV, 'w')
    outputLSV.write('accuracy of model: ')
    outputLSV.write('%.2f%%\n' % (accuracy_score(y_test, LSV_y_pred)*100))
    for review, cls in zip(X_test, LSV_y_pred):
        outputLSV.write('%s | %r\n' % (reviews.target_names[cls], review))
    

modelNaiveBayesAndSupportVector(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
