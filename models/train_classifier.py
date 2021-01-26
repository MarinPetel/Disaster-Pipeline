import sys
import pandas as pd
from sqlalchemy import create_engine

# import nlp tools
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    '''Load the databse and return the features and targets as dataframes

    INPUT:
    database_filepath-string, the path of the database to load_data

    OUTPUT:
    X-Dataframe, the dataframe with the original features
    Y-Dataframe, the dataframe with the targets
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql(sql='SELECT * FROM Disaster',con=engine)
    # need to filter out some messages which have value of 2 for the categorie
    # 'related'
    df = df[df['related']!=2]
    X = df['message']
    Y = df.drop(columns=['id','message','original','genre'],axis=1)
    return X,Y

def tokenize(text):
    pass


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
