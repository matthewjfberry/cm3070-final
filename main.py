import os
import tkinter as tk
from tkinter import ttk
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from nltk.tokenize import word_tokenize
import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle

# Create the main window
window = tk.Tk()
window.title("Fake News Checker")
window.resizable(False, False)

# Create a frame for the radio buttons
radio_frame = ttk.Frame(window)
radio_frame.pack(padx=10, pady=(10, 0))

# Create radio buttons for input type selection
input_type = tk.StringVar(value="url")
url_radio = ttk.Radiobutton(radio_frame, text="URL", variable=input_type, value="url")
url_radio.pack(side="left", padx=(0, 10))
text_radio = ttk.Radiobutton(radio_frame, text="Text", variable=input_type, value="text")
text_radio.pack(side="left")

# Create a frame for the input field and submit button
input_frame = ttk.Frame(window)
input_frame.pack(padx=10, pady=10)


def handle_focus_in(event):
    if input_entry.get("1.0", "end-1c") == tip_text:
        input_entry.delete("1.0", "end")
        input_entry.config(fg="black")


def handle_focus_out(event):
    if input_entry.get("1.0", "end-1c") == "":
        input_entry.insert("1.0", tip_text)
        input_entry.config(fg="gray")


def handle_input(event):
    if input_entry.get("1.0", "end-1c") == tip_text:
        input_entry.delete("1.0", "end")
        input_entry.config(fg="black")


# Create an input field for URL or text
tip_text = "Paste the news article URL or text here"
input_entry = tk.Text(input_frame, height=10, width=50)
input_entry.insert("1.0", tip_text)
input_entry.config(fg="gray")
input_entry.bind("<FocusIn>", handle_focus_in)
input_entry.bind("<FocusOut>", handle_focus_out)
input_entry.bind("<Key>", handle_input)

# Create a label to display the result
result_label = ttk.Label(window, text="", font=("Arial", 14))
result_label.pack(pady=10)

input_entry.pack()


def check_news():
    input_text = input_entry.get("1.0", "end-1c").strip()

    if input_text == tip_text:
        result_label.config(text="Please enter a valid URL or text.", foreground="black")
        return

    if input_type.get() == "url":
        try:
            page_text = fetch_page_text(input_text)
            if page_text:
                process_text(page_text)
        except requests.exceptions.RequestException as e:
            result_label.config(text="Error occurred while fetching the URL.", foreground="black")
            return
    else:
        process_text(input_text)


def fetch_page_text(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful
        response.raise_for_status()

        # Create a BeautifulSoup object to parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the main content of the page (you may need to adjust this based on the website structure)
        main_content = soup.find('body')

        if main_content:
            # Extract the text from the main content
            page_text = main_content.get_text(separator=' ')

            # Remove leading/trailing whitespace and normalize the text
            page_text = ' '.join(page_text.split())
            return page_text
        else:
            result_label.config(text="Main content not found on the page.", foreground="black")
            return None

    except requests.exceptions.RequestException as e:
        result_label.config(text="Error occurred while fetching the URL.", foreground="black")
        return None


def process_text(text):
    cleaned_text = clean_text(text)
    tokenized_text = word_tokenize(cleaned_text)
    text_no_stopwords = remove_stopwords(tokenized_text)
    lemmatized_text = lemmatize_text(text_no_stopwords)
    lemmatized_text_joined = ' '.join(lemmatized_text)

    # Transform the input text using the trained TfidfVectorizer
    input_tfidf = tfidf_vectorizer.transform([lemmatized_text_joined])

    # Predict the label using the trained PassiveAggressiveClassifier
    prediction = pac.predict(input_tfidf)

    # Update the label text and color based on the prediction result
    if prediction[0] == 0:
        result_label.config(text="Reliable", foreground="green")
    else:
        result_label.config(text="Unreliable", foreground="red")


# Create a submit button
submit_button = ttk.Button(input_frame, text="Check News", command=check_news)
submit_button.pack(pady=10)


def clean_text(text):
    # Check if the text is a string
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuations and numbers
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Single character removal
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    # Removing multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Convert to lowercase
    text = text.lower()
    return text


def remove_stopwords(text):
    return [word for word in text if word not in stop_words]


def lemmatize_text(text):
    return [lemmatizer.lemmatize(word) for word in text]


def display_sample(data, n=5):
    sample_indices = random.sample(range(len(data)), n)
    for i in sample_indices:
        print(f"Original Text:\n{data.loc[i, 'text']}\n")
        print(f"Cleaned Text:\n{data.loc[i, 'cleaned_text']}\n")
        print(f"Tokenized Text:\n{data.loc[i, 'tokenized_text']}\n")
        print(f"Text without Stopwords:\n{data.loc[i, 'text_no_stopwords']}\n")
        print(f"Lemmatized Text:\n{data.loc[i, 'lemmatized_text']}\n")
        print("----------------------------------------------------\n")


# Load the stop words
stop_words = set(nltk.corpus.stopwords.words('english'))

# Initialize the lemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()

# Initialize the TfidfVectorizer and PassiveAggressiveClassifier
tfidf_vectorizer = None
pac = None

# Check if the saved model and vectorizer files exist
if os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl'):
    # Load the saved model and vectorizer
    with open('model.pkl', 'rb') as file:
        pac = pickle.load(file)

    with open('vectorizer.pkl', 'rb') as file:
        tfidf_vectorizer = pickle.load(file)
else:
    # Load the training and testing data
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    # Apply text cleaning to the training data
    train_data['cleaned_text'] = train_data['text'].apply(clean_text)

    # Download the necessary NLTK data
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Tokenize the cleaned text
    train_data['tokenized_text'] = train_data['cleaned_text'].apply(word_tokenize)

    # Remove stop words from the tokenized text
    train_data['text_no_stopwords'] = train_data['tokenized_text'].apply(remove_stopwords)

    # Lemmatize the text without stop words
    train_data['lemmatized_text'] = train_data['text_no_stopwords'].apply(lemmatize_text)

    # Join the lemmatized text
    train_data['lemmatized_text_joined'] = train_data['lemmatized_text'].apply(lambda x: ' '.join(x))

    # Initialize and train the model and vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    tf_idf_matrix = tfidf_vectorizer.fit_transform(train_data['lemmatized_text_joined'])

    y_df = train_data['label']
    x_train, x_test, y_train, y_test = train_test_split(tf_idf_matrix, y_df, random_state=0)

    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(x_train, y_train)

    y_pred = pac.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Passive-Aggressive Classifier Accuracy: {score * 100:.2f}%')

    # Save the trained model and vectorizer
    with open('model.pkl', 'wb') as file:
        pickle.dump(pac, file)

    with open('vectorizer.pkl', 'wb') as file:
        pickle.dump(tfidf_vectorizer, file)

# Start the main event loop
window.mainloop()