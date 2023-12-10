import streamlit as st
import pickle
import nltk
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Routes
#@app.route('/')
def welcome():
    return "Welcome All"

def hindi_tokenizer(text):
    tokens = nltk.word_tokenize(text, language='hindi', preserve_line=True)
    
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    
    # remove punctuation from each word
    words = [re_punc.sub('', w) for w in tokens]
    return words
# global variables
tfidf = TfidfVectorizer(tokenizer=hindi_tokenizer)
svm = SVC()
lbl = LabelEncoder()

# Replacing this with your actual training data and model loading code
with open('tfidf_vectorizer1.pkl', 'rb') as file:
    tfidf = pickle.load(file)
with open('model1.pkl', 'rb') as file:
    svm = pickle.load(file)
with open('label_encoder1.pkl', 'rb') as file:
    lbl = pickle.load(file)


#@app.route('/predict', methods=["GET"])
def predict_emails(text):
    txt_vect = tfidf.transform([text])
    prediction = svm.predict(txt_vect)
    return f"The Mail {text} is: {lbl.inverse_transform(prediction)[0]}."

# Web page part use H1 format
def main():
    st.title("Email Spam/Ham Classification")
    st.write("Enter an email below to classify if it's spam or ham!")

    html_temp = """<div style="background-color:#25246; padding:10px">
    <h2 style="color:white; text-align:center;">Spam Email Classification</h2>
    </div>
    """
    
    st.markdown(html_temp, unsafe_allow_html=True)  # To render the html code as html
    
    # Getting the input from the user
    input_text = st.text_input("Enter the message")
    
    spam_html = """
    <div style="background-color:#F4D03F; padding:10px">
    <h2 style="color:white; text-align:center;">This Email is Spam</h2>
    </div>
    """
    
    ham_html = """
    <div style="background-color:#F4D03F; padding:10px">
    <h2 style="color:white; text-align:center;">This Email is Ham</h2>
    </div>
    """
    
    if st.button("Click to predict"):
        output = predict_emails(input_text)
        
        st.success("The prediction is {}".format(output))
        
        if output == 1:
            st.markdown(spam_html, unsafe_allow_html=True)
        else:
            st.markdown(ham_html, unsafe_allow_html=True)

# The block to run the Streamlit app
if __name__ == '__main__':
    main()
