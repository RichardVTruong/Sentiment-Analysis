# Importing libraries and functions.
import contractions
import numpy as np
import pandas as pd
from autocorrect import Speller
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Defining global variables.
tfidf_vectorizer = None
model = None

# Defining the "clean" function.
# Used to:
# 1) Lowercase the contents of the document.
# 2) Expand contracted words.
# 3) Tokenize the document.
# 4) Remove stop words and returns non-puncutation characters.

def clean(document):
    document['reviews'] = document['reviews'].str.lower()
    spell = Speller(lang='en')
    document_corrected = spell(document)
    document_tokenized = word_tokenize(document_corrected)
    document_expanded = contractions.fix(word for word in document_tokenized)
    stop_words = stopwords.words('English')
    document_no_stopwords = ' '.join(
        [word for word in document_expanded if word not in stop_words and document_expanded.isalnum()])
    document = document_no_stopwords

    return document

def train_model():
    global model
    global tfidf_vectorizer

    df1_original = pd.read_table('imdb_labelled.txt', names=['reviews', 'sentiment'])
    df2_original = pd.read_table('yelp_labelled.txt', names=['reviews', 'sentiment'])
    df3_original = pd.read_table('amazon_cells_labelled.txt', names=['reviews', 'sentiment'])

    df1 = df1_original.copy()
    df2 = df2_original.copy()
    df3 = df3_original.copy()
    df4 = pd.concat([df1, df2, df3], ignore_index=True)

    clean(df4['reviews'])

    tfidf_vectorizer = TfidfVectorizer()
    data_numeric_form = tfidf_vectorizer.fit_transform(df4['reviews'])

    x_train, x_test, y_train, y_test = train_test_split(data_numeric_form.todense(), df4['sentiment'])

    log_reg = LogisticRegression()
    model = log_reg.fit(np.asarray(x_train), y_train)

    accuracy = log_reg.score(np.asarray(x_test), y_test)

    print("Logistic Regression accuracy: ", accuracy)

def predict(review):
    global model
    global tfidf_vectorizer

    clean(review)

    review_numeric_form = tfidf_vectorizer.transform(review)

    prediction = model.predict(review_numeric_form)

    print("The prediction for review: ", review, "is: ", prediction)

def testing():
    train_model()
    predict("This item is great!")

testing()
