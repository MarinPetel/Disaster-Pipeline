import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load the two csv files containing messages and categories_filepath

    INPUT:
    messages_filepath-string, the path to the csv file containing messages
    categories_filepath-string, the path to the csv file containing messages
    categories

    OUTPUT:
    df-Dataframe, merged dataframe containing the messages and their categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #'id' is the common key in both files and has the same name
    df = pd.merge(left=messages,right=categories,on='id')

    return df


def clean_data(df):
    '''
    Returns a cleaned version of the merged Dataframe

    INPUT:
    df-Dataframe, the merged dataframe

    OUTPUT:
    df-Dataframe, cleaned dataframe with descriptive column names for categories
    , binary variables for the categories and non duplicated rows.
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = [x[:-2] for x in row]
    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert string values to binary variables
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(np.int8)

    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # need to filter out some messages which have value of 2 for the categorie
    # 'related'
    df = df[df['related']!=2]

    return df


def save_data(df, database_filename):
    '''
    Saves a dataframe as a sql database

    INPUT:
    df-Dataframe, the dataframe to be saved
    database_filename-string, the name to give to the saved sql database

    OUTPUT:
    database_filename,sql database
    '''
    engine = create_engine('sqlite:///' + database_filename + '.db')
    df.to_sql(database_filename, engine, index=False, if_exists = 'replace')


def main():
    '''
    Runs all the above functions after having taken the files paths and name for
    the new sql database
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
