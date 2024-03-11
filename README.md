# Fake News Checker
This is a Python-based Fake News Checker application that allows users to determine the reliability of news articles by analyzing their content. The application provides a graphical user interface (GUI) built with Tkinter, allowing users to input either a URL or the text of a news article for analysis.

## Features
- Input news articles via URL or direct text input
- Fetches the main content of a webpage when provided with a URL
- Cleans and preprocesses the text data using techniques such as HTML tag removal, punctuation removal, tokenization, stop word removal, and lemmatization
- Utilizes a pre-trained PassiveAggressiveClassifier model and TfidfVectorizer for text classification
- Displays the reliability of the news article as either "Reliable" or "Unreliable"
- Saves the trained model and vectorizer for future use

## Requirements
- Python 3.x
- Tkinter
- requests
- BeautifulSoup
- pandas
- nltk
- scikit-learn

## Dataset

The application relies on a pre-trained model that was trained using the `train.csv` and `test.csv` datasets. These datasets should be placed in the same directory as the application.

## Model Training

If the pre-trained model and vectorizer files (`model.pkl` and `vectorizer.pkl`) are not found, the application will automatically train a new model using the provided dataset. The trained model and vectorizer will be saved for future use.

## Acknowledgements
- The PassiveAggressiveClassifier and TfidfVectorizer are part of the scikit-learn library.
- The text preprocessing techniques are implemented using the Natural Language Toolkit (NLTK).
- The Fake news dataset was obtained from [Kaggle.com (ELMAHALAWY, M - Fake news - Kaggle.com - 2023)](https://www.kaggle.com/datasets/marwanelmahalawy/fake-news?resource=download)

## License

This project is licensed under the MIT License.
