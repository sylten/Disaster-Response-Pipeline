# Disaster Response Pipeline Project
Machine learning model that attempts to classify short messages into disaster response related categories. The project includes a simple web app that can classify new messages into these categories.

The dataset is however quite imbalanced which makes classifying tough. The web app includes a plot of the number of occurances of each category to visualize the imbalance. The app also provides a view of the words included in messages tagged with each category. Looking at this word data we notice for example that in some cases words that seem unrelated to the category are among the most common, this may lead the model astray.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        *Note: there is a checked in classifier model in models/classifier.pkl. Training takes about 10 minutes so it's included for ease of use.*

2. Run the following command in the app directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

    - Messages are classified by entering into the text input and clicking the Classify Message button.
    
    - Use the select-box to view the most common words in training messages that belong to the selected category.

### Files:
**app/run.py**
Flask web server. Serves web app with template files found in app/templates.

**data/process_data.py**
Runs ETL pipeline that cleans and stores data from csv files into SQLite database.

**models/train_classifier.py**
Runs machine learning pipeline, which trains a classifier using data from the database created by process_data.py.

**models/classifier.pkl**
Training the model takes around 10 minutes, this saved model is checked in for ease of use.
