import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView

class FakeNewsDetectorApp(App):
    def build(self):
        # Read the data
        df = pd.read_csv(r'C:\Users\theam\Downloads\news\news.csv')

        # Separate features and labels
        x = df['text']
        y = df['label']

        # Split the dataset into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

        # Transform the training and testing sets
        tfidf_train = self.tfidf_vectorizer.fit_transform(x_train)
        tfidf_test = self.tfidf_vectorizer.transform(x_test)

        # Create and train the PassiveAggressiveClassifier
        self.pac = PassiveAggressiveClassifier(max_iter=50)
        self.pac.fit(tfidf_train, y_train)

        # Evaluate the model on the test set
        y_pred = self.pac.predict(tfidf_test)
        score = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {round(score * 100, 2)}%')

        # Display confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
        print('Confusion Matrix:')
        print(conf_matrix)

        self.user_input_label = Label(text='Enter a news article text:')
        self.user_input_text = TextInput(multiline=True)
        self.predict_button = Button(text='Predict')
        self.predict_button.bind(on_press=self.predict_article)

        self.result_label = Label(text='Prediction:')
        self.result_text = TextInput(readonly=True, multiline=True, height=100, size_hint_y=None)

        layout = BoxLayout(orientation='vertical')
        scroll_view = ScrollView(size=(300, 500))
        scroll_view.add_widget(layout)

        layout.add_widget(self.user_input_label)
        layout.add_widget(self.user_input_text)
        layout.add_widget(self.predict_button)
        layout.add_widget(self.result_label)
        layout.add_widget(self.result_text)

        return scroll_view

    def predict_article(self, instance):
        user_input = self.user_input_text.text

        # Transform the user input using the same TF-IDF vectorizer
        tfidf_user_input = self.tfidf_vectorizer.transform([user_input])

        # Make prediction
        prediction = self.pac.predict(tfidf_user_input)

        # Display result
        self.result_text.text = f'Prediction: {prediction[0]}'

if __name__ == '__main__':
    FakeNewsDetectorApp().run()
