import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
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
    df = df.drop_duplicates(keep=False) 
    df = df.fillna(0)

    return df


def save_data(df, database_filename):
    engine = create_engine("sqlite:///" + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists="replace")
    pass  


def main():
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
    #main()
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    df = pd.read_sql_table('Messages', engine)
    # df[df[['id', 'message', 'original', 'genre']]]
    # print(df.drop(columns=['id', 'message', 'original', 'genre']).sum())
    # genre_counts = df.groupby('genre').count()['message']
    # genre_names = list(genre_counts.index)
    # print(genre_counts)

    # word_list = 
    # word_counter = {}
    # for word in word_list:
    #     if word in word_counter:
    #         word_counter[word] += 1
    #     else:
    #         word_counter[word] = 1
    
    # popular_words = sorted(word_counter, key = word_counter.get, reverse = True)
    
    # top_3 = popular_words[:3]
    
    # print(top_3)

    with engine.connect() as con:
        rs = con.execute("select GROUP_CONCAT(message, ' ') from messages")
        import re
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        text = rs.next()[0]
        text = re.sub(f"[^A-Za-z]", " ", text).lower()
        word_list = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        word_list = [w for w in word_list if not w in stop_words] 

        word_counter = {}
        for word in word_list:
            if word in word_counter:
                word_counter[word] += 1
            else:
                word_counter[word] = 1
        
        # popular_words = sorted(word_counter, key = word_counter.get, reverse = True)
        popular_words = sorted(word_counter.items(), key=lambda tup: tup[1], reverse=True)

        top_3 = popular_words[:10]
        
        print(top_3)