import json
import plotly
import pandas as pd
import sys
import os
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

sys.path.append(os.path.abspath("../models"))
from train_classifier import tokenize

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")

def get_top_words(category = None, top_n = 10):
    '''
    Description: finds the most common words used in messages flagged with a certain category.

    Arguments:
        category (string): Category of messages to count words in. Default=None (count all categories)
        top_n (int): Number of top results to return

    Returns:
        list: List of (category, count) tuples.
    '''

    # query all messages from database
    messages_df = df
    if category:
        messages_df = messages_df[messages_df[category] == 1]

    text = messages_df.message.str.cat(sep=" ")
    
    # remove special characters
    text = re.sub(f"[^A-Za-z]", " ", text).lower()

    word_list = word_tokenize(text)

    # remove stop words
    stop_words = set(stopwords.words('english'))
    word_list = [w for w in word_list if not w in stop_words] 

    # count occurance of words
    word_counter = {}
    for word in word_list:
        if word in word_counter:
            word_counter[word] += 1
        else:
            word_counter[word] = 1
    
    # sort by word count
    sorted_word_counts = sorted(word_counter.items(), key=lambda tup: tup[1], reverse=True)

    return sorted_word_counts[:top_n]

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    category = request.args.get('category', '') 

    # extract data needed for visuals
    top_words = get_top_words(category)
    word_counts = []
    words = []
    for w in top_words:
        words.append(w[0])
        word_counts.append(w[1])

    words_title = 'Most common words'
    if category:
        words_title += ' for category ' + category

    categories_counts = df.drop(columns=['id', 'message', 'original', 'genre']).sum().sort_values()
    categories = list(categories_counts.index)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=words,
                    y=word_counts,
                )
            ],

            'layout': {
                'title': words_title,
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Words"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories_counts,
                    y=categories,
                    orientation='h',
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Category"
                },
                'xaxis': {
                    'title': "Count"
                },
                'height': 1200
            },
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON, categories=categories)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()