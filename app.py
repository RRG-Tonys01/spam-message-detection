import numpy as np  
import pandas as pd  
import tensorflow as tf 
from sklearn.model_selection import train_test_split 
from tensorflow.keras.models import Sequential, load_model  
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from flask import Flask, render_template, request, jsonify 

import warnings 
warnings.filterwarnings("ignore")



class SpamDetector:
    # Load and preprocess the CSV file using Pandas
    def __init__(self, csv_file, model_save_path):
        self.df = pd.read_csv(csv_file, encoding='latin-1')
        self.df = self.df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
        self.df = self.df.rename(columns={'v1': 'label', 'v2': 'Text'})
        self.df['label_enc'] = self.df['label'].map({'ham': 0, 'spam': 1})
        self.avg_words_len = round(sum([len(i.split()) for i in self.df['Text']]) / len(self.df['Text']))
        self.s = set()
        for sent in self.df['Text']:
            for word in sent.split():
                self.s.add(word)
        self.total_words_length = len(self.s)
        self.MAXTOKENS = self.total_words_length
        self.OUTPUTLEN = self.avg_words_len
        self.X, self.y = np.asanyarray(self.df['Text']), np.asanyarray(self.df['label_enc'])
        self.new_df = pd.DataFrame({'Text': self.X, 'label': self.y})
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.new_df['Text'], self.new_df['label'], test_size=0.2, random_state=42)
        self.X_train_seq, self.X_train_pad, self.X_test_seq, self.X_test_pad = self.tokenize_and_pad()
        self.model_save_path = model_save_path
        self.model = self.load_or_train_model()
        self.tokenizer = self.create_tokenizer()

    def tokenize_and_pad(self):
        tokenizer = Tokenizer(num_words=self.MAXTOKENS, oov_token="<OOV>")
        tokenizer.fit_on_texts(self.X_train)

        X_train_seq = tokenizer.texts_to_sequences(self.X_train)
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.OUTPUTLEN, padding='post', truncating='post')

        X_test_seq = tokenizer.texts_to_sequences(self.X_test)
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.OUTPUTLEN, padding='post', truncating='post')

        return X_train_seq, X_train_pad, X_test_seq, X_test_pad

    def create_tokenizer(self):
        tokenizer = Tokenizer(num_words=self.MAXTOKENS, oov_token="<OOV>")
        tokenizer.fit_on_texts(self.X_train)
        return tokenizer

    def load_or_train_model(self):
        try:
            model = load_model(self.model_save_path)
        except (OSError, IOError):
            model = self.build_model()
            model.save(self.model_save_path)
        return model

    def build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.MAXTOKENS, output_dim=64, input_length=self.OUTPUTLEN))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(self.X_train_pad, self.y_train, epochs=5, validation_data=(self.X_test_pad, self.y_test))
        return model

class FlaskApp:
    def __init__(self, spam_detector):
        self.app = Flask(__name__, template_folder='templates')
        self.spam_detector = spam_detector
        self.register_routes()

    def register_routes(self):
        @self.app.route("/")
        def home():
            return render_template('index.html')

        @self.app.route("/message", methods=['POST'])
        def check_message_input():
            input_message = request.form.get('message')
            sample_sentence = input_message
            sample_seq = self.spam_detector.tokenizer.texts_to_sequences([sample_sentence])
            sample_pad = pad_sequences(sample_seq, maxlen=self.spam_detector.OUTPUTLEN, padding='post',
                                       truncating='post')
            prediction = self.spam_detector.model.predict(sample_pad)
            if prediction[0][0] >= 0.5:
                message = 'Its A Spam'
            else:
                message = 'Its A Ham'

            response_data = {'message': message}
            return jsonify(response_data)

    def run_app(self):
        if __name__ == "__main__":
            self.app.run(debug=True)



model_save_path = "spam_model.h5"
spam_detector = SpamDetector("spam-dataset.csv", model_save_path)
flask_app = FlaskApp(spam_detector)
flask_app.run_app()