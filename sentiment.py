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


# Defining the "clean" function. Used to lowercase the entire contents of the document, expands contracted words, spell checks the words, 

def clean(document):
    # TODO: implement your document preprocessing tasks:
    #   -> Lowercase the document
    #   -> Replace "hadn't" with "had not"
    #   -> Replace "wasn't" with "was not"
    #   -> Replace "didn't" with "did not"
    #   -> Spell check
    #   -> Remove stop words. You will need to tokenize the document and remove stop words
    #   ->   at this step you should also remove punctuation.
    #        Tip: use the same method as shown in class to remove stopwords, but you'll need
    #             1 more condition. Not only should the token in the document not be in the
    #             the stop words, but it should also be a non-punctuation character. (use isalnum())

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
    # 1. TODO Read data: amazon_cells_labeled.txt
    #               imdb_labelled.txt
    #               yelp_labelled.txt
    #    Tip: Use Pandas readTable method to read in a single txt into a dataframe (as shown in Week 9)
    #    Tip: Use Pandas concat (pd.concat) to combine all three dataframes into 1.
    df1_original = pd.read_table('imdb_labelled.txt', names=['reviews', 'sentiment'])
    df2_original = pd.read_table('yelp_labelled.txt', names=['reviews', 'sentiment'])
    df3_original = pd.read_table('amazon_cells_labelled.txt', names=['reviews', 'sentiment'])

    df1 = df1_original.copy()
    df2 = df2_original.copy()
    df3 = df3_original.copy()

    df4 = pd.concat([df1, df2, df3], ignore_index=True)

    # 2. TODO For the reviews, apply the clean function above to each document

    clean(df4['reviews'])

    # 3. TODO Convert each review to its numerical form using TF-IDF (as shown in class).
    #    Set the TF-IDF vectorizer to the model tfidf_vectorizer so that you can use it
    #    outside of this function, in particular, you'll be using it inside the prediction
    #    function to transform new incoming reviews into their numeric forms.

    tfidf_vectorizer = TfidfVectorizer()
    data_numeric_form = tfidf_vectorizer.fit_transform(df4['reviews'])

    # 4. TODO Split the dataset into train/test sections (as shown in class)

    x_train, x_test, y_train, y_test = train_test_split(data_numeric_form.todense(), df4['sentiment'])

    # 5. TODO Train a logistic Regression Model, and set the model to point to the global model
    #    variable so that you can access it outside of this function.

    log_reg = LogisticRegression()
    model = log_reg.fit(np.asarray(x_train), y_train)

    # 6. TODO Use the test dataset to get the accuracy of your model

    accuracy = log_reg.score(np.asarray(x_test), y_test)

    print("Logistic Regression accuracy: ", accuracy)


def predict(review):
    global model
    global tfidf_vectorizer
    # 1. TODO Clean the review as you did your dataset. (by using the clean function)

    clean(review)

    # 2. TODO Using the vectorizer that was fit_transformed in train_model, and stored in the
    #    global variable tfidf_vectorizer to transform the review into its numeric form.

    review_numeric_form = tfidf_vectorizer.transform(review)

    # 3. TODO Call the model that was trained, and stored in the global variable model to
    #    predict the result of the model.

    prediction = model.predict(review_numeric_form)

    print("The prediction for review: ", review, "is: ", prediction)


def testing():
    train_model()
    predict("This item is great!")


testing()
