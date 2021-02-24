import time
import joblib
import nltk
import numpy as np
nltk.download(['punkt', 'stopwords'])

import sys
import pandas as pd
import re
import pickle
import string
from tabulate import tabulate
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

debug_tokenize = False

def load_data(database_filepath):
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table('Messages', con = engine)
    # df = df[100:200]
    
    X = df.message.values

    categories_df = df.drop(columns=['id', 'message', 'original', 'genre'], axis=1)
    categories_df = categories_df.fillna(0)
    Y = categories_df.values
    category_names = categories_df.columns.values

    return X, Y, category_names

def remove_named_entities(tokens):
    tagged = nltk.pos_tag_sents([tokens])
    # print(tagged)
    chunked = nltk.ne_chunk_sents(tagged)
    # print(chunked)
    def extract_nonentities(tree):
        return [leaf[0] for leaf in tree if type(leaf) != nltk.Tree]
    
    return extract_nonentities(next(chunked))

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def tokenize(text):
    # remove non-letter characters
    filtered_text = re.sub(f"[^A-Za-z]", " ", text).lower()

    # remove single characters
    filtered_text = re.sub(r'\s+[a-zA-Z]\s+', ' ', filtered_text)

    tokens = word_tokenize(filtered_text)
    
    if debug_tokenize:
        print(tokens)

    # tokens = remove_named_entities(tokens)

    if debug_tokenize:
        print(tokens)
    
    # convert to lowercase, (done on tokens because it needs to be done after removing named entities)
    # tokens = [t.lower() for t in tokens]
    
    # remove stop words
    filtered_tokens = [w for w in tokens if not w in stop_words] 
    
    if debug_tokenize:
        print(filtered_tokens)
    
    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in filtered_tokens]
    
    if debug_tokenize:
        print()
    
    return lemmatized_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer()),

        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    parameters = {
        'vect__max_features': (1000, 10000),
        'vect__min_df': [1, 2, 5],
        # 'tfidf__use_idf': (True, False),  
        # 'clf__estimator__alpha': [1, 5],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    
    return cv


def evaluate_model(model, x_test, y_test, category_names):
    print(model.best_params_)

    y_pred = model.predict(x_test)
    print(y_test.shape, y_pred.shape)
    # y_test = np.vstack(y_test)
    # y_pred = np.vstack(y_pred)

    # rows = []
    # for ix, column in enumerate(category_names):
    #     column_y_test = y_test[:,ix]
    #     column_y_pred = y_pred[:,ix]
    #     rows.append([
    #         column, 
    #         round(f1_score(column_y_test, column_y_pred, average='macro'), 2),
    #         round(precision_score(column_y_test, column_y_pred, average='macro'), 2),
    #         round(recall_score(column_y_test, column_y_pred, average='macro'), 2)
    #     ])

    # print(tabulate(rows, headers=['Category', 'F1-score', 'Precision', 'Recall'], tablefmt='orgtbl'))
    # print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
    print(classification_report(y_test, y_pred, target_names=category_names))
    print(accuracy_score(y_test, y_pred))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=3)
        
        print(len(X))
        if debug_tokenize:
            start_time = time.time()
            for i in range(10):
                tokenize(X[i])
            print('  Tokenizing took',round((time.time()-start_time), 10), 'seconds')
            exit()

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        start_time = time.time()
        model.fit(X_train, Y_train)
        print('  Training took',round((time.time()-start_time), 2), 'seconds')

        #model = joblib.load(model_filepath)
                
        print('Evaluating model...')
        evaluate_model(model, x_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()