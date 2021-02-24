import time
import nltk
nltk.download(['punkt', 'stopwords'])

import sys
import pandas as pd
import re
import pickle
from tabulate import tabulate
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

debug_tokenize = False

def load_data(database_filepath):
    '''
    Description: Load disaster response messages and categories from SQLite database.

    Arguments:
        database_filepath (string): File path to databse to load from.

    Returns:
        list: List of messages
        list: List of lists of category values
        list: List of categories
    '''

    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table('Messages', con = engine)
    # df = df[100:200]

    categories_df = df.drop(columns=['id', 'message', 'original', 'genre'], axis=1)
    categories_df = categories_df.fillna(0)
    category_names = categories_df.columns.values

    X = df.message.values
    Y = categories_df.values

    return X, Y, category_names

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def tokenize(text):
    '''
    Description: Removes special characters and stop words, tokenizes and lemmetizes text.

    Arguments:
        text (string): Text to process.

    Returns:
        string: Tokenized text.
    '''

    # remove non-letter characters
    filtered_text = re.sub(f"[^A-Za-z]", " ", text).lower()

    # remove single characters
    filtered_text = re.sub(r'\s+[a-zA-Z]\s+', ' ', filtered_text)

    tokens = word_tokenize(filtered_text)
    
    if debug_tokenize:
        print(tokens)

    if debug_tokenize:
        print(tokens)
    
    # remove stop words
    filtered_tokens = [w for w in tokens if not w in stop_words] 
    
    if debug_tokenize:
        print(filtered_tokens)
    
    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in filtered_tokens]
    
    if debug_tokenize:
        print()
    
    return lemmatized_tokens


def build_model():
    '''
    Description: Create machine learning model to classify text.

    Returns:
        GridSearchCV: Grid search optomized model
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer()),

        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    parameters = {
        'vect__max_features': (1000, 10000),
        'vect__min_df': [1, 2, 5]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    
    return cv


def evaluate_model(model, x_test, y_test, category_names):
    '''
    Description: Evaluates and prints machine learning model's precision, recall and f1-score for each class. As well as accuracy.

    Arguments:
        model: Model to evaluate
        x_test: Test list of tokenized messages
        y_test: List of true test categories
        category_names: Names of test categories

    Returns:
        None
    '''

    print(model.best_params_)

    y_pred = model.predict(x_test)

    print(classification_report(y_test, y_pred, target_names=category_names))
    print(accuracy_score(y_test, y_pred))


def save_model(model, model_filepath):
    '''
    Description: Save model as pickle file.

    Arguments:
        model: Model to save
        model_filepath: Path to save file to

    Returns:
        None
    '''

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    Description: Loads data from database, creates model, evaluates and saves to pickle file.

    Returns:
        None
    '''

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