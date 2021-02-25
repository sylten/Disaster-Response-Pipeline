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

    # read messages and categories
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge into single dataframe
    df = pd.merge(messages, categories, on="id")

    def split_keyvalue(keyvalue):
        '''
        Description: Split string of format key-value into string and binary integer value

        Arguments:
            keyvalue (string): String to split

        Returns:
            key: The key part
            value: The binary value part
        '''
        key, value = keyvalue.split('-', 1)
        return key, 1 if value == "1" else 0

    # categories are all contained in a single string of the format category1-value1;category2-value2
    # split the string into a df with categories as columns
    categories_df = pd.DataFrame.from_records([
        dict(split_keyvalue(keyvalue) for keyvalue in category) 
            for category in df.categories.str.split(';')
    ])

    # remove the unformated categories column
    df = df.drop(columns=['categories'], axis=1)

    # add the new category columns
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