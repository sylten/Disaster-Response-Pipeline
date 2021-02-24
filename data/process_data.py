import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Description: Loads disaster response messages from csv files and merges them into a data frame.

    Arguments:
        messages_filepath (string): Path to messages csv file.
        categories_fiepath (string): Path to categorues csv file.

    Returns:
        DataFrame: Merged dataframe
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages, categories, on="id")

    def split_keyvalue(keyvalue):
        key, val = keyvalue.split('-', 1)
        return key, 1 if val == "1" else 0

    categories_df = pd.DataFrame.from_records([
        dict(split_keyvalue(keyvalue) for keyvalue in category) 
            for category in df.categories.str.split(';')
    ])

    df = df.drop(columns=['categories'], axis=1)

    df = pd.concat([df, categories_df], axis=1)

    return df


def clean_data(df):
    '''
    Description: Drops duplicates.

    Arguments:
        df (DataFrame): dataframe to clean. 

    Returns:
        DataFrame: Cleaned dataframe
    '''

    df = df.drop_duplicates(keep=False) 
    df = df.fillna(0)

    return df


def save_data(df, database_filename):
    '''
    Description: Saves dataframe to SQLite database, overwriting existing data.

    Arguments:
        df (DataFrame): dataframe to save.
        database_filename (string): File path to databse to save to.

    Returns:
        None
    '''

    engine = create_engine("sqlite:///" + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists="replace")


def main():
    '''
    Description: Loads disaster response messages and categories from csv files, cleans and saves to database.

    Returns:
        None
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