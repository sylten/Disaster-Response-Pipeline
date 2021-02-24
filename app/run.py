import json
import plotly
import pandas as pd
import sys
import os
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
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

def get_top_words(label = 'cold', top_n = 10):
    # query all messages from database
    messages_df = df
    if label:
        messages_df = messages_df[messages_df[label] == 1]

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

    label = request.args.get('label', '') 

    # extract data needed for visuals
    top_words = get_top_words(label)
    word_counts = []
    words = []
    for w in top_words:
        words.append(w[0])
        word_counts.append(w[1])

    words_title = 'Most common words'
    if label:
        words_title += ' for label ' + label

    labels_counts = df.drop(columns=['id', 'message', 'original', 'genre']).sum().sort_values()
    labels = list(labels_counts.index)

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
                    x=labels_counts,
                    y=labels,
                    orientation='h',
                )
            ],

            'layout': {
                'title': 'Distribution of Message Labels',
                'yaxis': {
                    'title': "Label"
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
    return render_template('master.html', ids=ids, graphJSON=graphJSON, labels=labels)


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