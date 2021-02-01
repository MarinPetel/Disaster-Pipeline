import sys
import pandas as pd
from sqlalchemy import create_engine
import os
import pickle
import matplotlib.pyplot as plt

# import nlp tools
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# imports from sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier,ClassifierChain
from sklearn.metrics import classification_report,accuracy_score,f1_score

def load_data(database_filepath):
    '''Load the databse and return the features, targets as dataframes and the
    target names

    INPUT:
    database_filepath-string, the path of the database to load_data

    OUTPUT:
    X-Dataframe, the dataframe with the original features
    Y-Dataframe, the dataframe with the targets
    category_names-array, the names of the targets
    '''
    # Read the SQL database
    engine = create_engine('sqlite:///' + database_filepath)
    print(database_filepath)
    db_name = os.path.basename(database_filepath).replace('.db','')
    df = pd.read_sql_table(db_name,con=engine)

    # Select the translated message as the original feature
    X = df['message']

    # Only keeps the binary variables with categories as the targets
    Y = df.drop(columns=['id','message','original','genre'],axis=1)
    category_names = Y.columns
    return X,Y,category_names

def tokenize(text):
    '''Process the text data in a message: normalization, tokenization
    and lemmatization

    INPUT:
    text-string, a message contained in the dataframe with features

    OUTPUT:
    text-list, a list with the normalized and lemmatized token words from the
    inputted message
    '''
    # find url and put a place holder
    url_regex=r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text=re.sub(url_regex,'urlplaceholder',text)

    # normalize:lowercase, remove punctuation
    text = text.lower().strip()
    text = re.sub(r'^a-zA-Z0-9',' ',text)

    # split into token words
    text = word_tokenize(text)

    # remove stop words
    text = [word for word in text if word not in stopwords.words('english')]

    # lemmatize
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]

    return text


def build_model():
    '''Builds a pipeline to process the data with a final estimator and combines
    it with a grid search. The accuracy will be maximized taking into account for
    one sample all the labels to be predicted. This is a tough metric as to be
    considered accurate a prediction for one message needs to be correct for all
    categories. When testing various models, the use of chainclassifier provided
    better results that MultiOutputClassifier which fits classifier for each category
    independantly of each other.

    OUTPUT:
    GridSearchCV- a GridSearchCV object to a pipeline
    '''
    pipeline = Pipeline([
    ('bag_of_words',CountVectorizer(tokenizer=tokenize)),
    ('tdif',TfidfTransformer()),
    ('ClassifierChain',ClassifierChain(AdaBoostClassifier()))
    ])

    parameters={'ClassifierChain__base_estimator__n_estimators':[100,150,200],
    'ClassifierChain__base_estimator__learning_rate':[1,0.8,0.5]}

    return GridSearchCV(pipeline,parameters,verbose=3)



def evaluate_model(model, X_test, Y_test, category_names):
    '''Evaluate the pipeline'''

    # Perform predictions
    predictions = model.predict(X_test)

    # Construct a dictionnary with accuracy and f1 scores for each categories
    d={}
    for category in category_names:
        d[category]={'accuracy':accuracy_score(Y_test[category],predictions[:,Y_test.columns.get_loc(category)]),
        'f1':f1_score(Y_test[category],predictions[:,Y_test.columns.get_loc(category)])}
    # Convert it in a dataframe, plot it and save as a png file
    pd.DataFrame(d).plot(kind='bar')
    plt.title('Accuracy and F1 Scores for each categories')
    plt.legend(loc=(-0.7,-0.7),ncol=8)
    plt.savefig('results',dpi=600,bbox_inches='tight')


    # Print classification report for each categories
    for category in category_names:
        print(f'Category {category}')
        print(classification_report(Y_test[category],
        predictions[:,Y_test.columns.get_loc(category)]))




def save_model(model, model_filepath):
    # Save the model as a pickle file
    pickle.dump(model, open(model_filepath, 'wb',-1))



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
        print('The best model has the following parameters:')
        print(model.best_params_)

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
