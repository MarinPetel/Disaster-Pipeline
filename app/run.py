import json
import plotly
import pandas as pd
import os

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
# Find list of files in the data folder
data_files = os.listdir('../data')
# Find the .db file
db_name = [x for x in data_files if x.endswith('.db')][0]
engine = create_engine('sqlite:///../data/' + db_name)
df = pd.read_sql_table(db_name.replace('.db',''), engine)

# load model
# Find list of files in the data folder
data_files = os.listdir('../models')
# Find the .pkl file
model_name = [x for x in data_files if x.endswith('.pkl')][0]
model = joblib.load("../models/" + model_name)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for  first visual
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # extract data needed for second visual
    genre_cat=df.groupby(by='genre')[df.columns[4:]].sum().T
    genre_cat['direct']=(genre_cat['direct']/genre_counts['direct'])*100
    genre_cat['social']=(genre_cat['social']/genre_counts['social'])*100
    genre_cat['news']=(genre_cat['news']/genre_counts['news'])*100

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Heatmap(
                    x=genre_names,
                    y=df.columns[4:],
                    z=genre_cat,
                    type='heatmap',
                    colorscale= 'ylgnbu'
                )
            ],

            'layout': {
                'title': 'Message Genres and Categories %',
                'yaxis': {
                    'title': "Categories"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


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
