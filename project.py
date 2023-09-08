#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# In[2]:


df = pd.read_csv("./final_dataset1.csv")


# # Pre-Processing

# In[3]:


lbl = LabelEncoder()
y = lbl.fit_transform(df['label'])


# In[4]:


x = df['Hindi']


# In[5]:


def hindi_tokenizer(text):
    tokens = word_tokenize(text,language='hindi',preserve_line=True)
    
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    
    # remove punctuation from each word
    words = [re_punc.sub('',w) for w in tokens]
    return words


# In[6]:


tfidf = TfidfVectorizer(tokenizer=hindi_tokenizer)
x_vect = tfidf.fit_transform(x)


# In[7]:


smote = SMOTE(random_state=0)


# In[8]:

x_sm, y_sm = smote.fit_resample(x_vect,y)

# In[11]:


x_train, x_test, y_train, y_test = train_test_split(x_sm,y_sm,test_size=0.3,random_state=24)


# In[17]:



mnb = MultinomialNB()
mnb.fit(x_train,y_train)

# In[ ]:


import streamlit as st


#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_emails(text):
    txt_vect = tfidf.transform([text])
    prediction = mnb.predict(txt_vect)
    return f"The Mail {text} \nis: {lbl.inverse_transform(prediction)[0]}."


def main():

    st.title("Email Spam/Ham Classification")
    st.write("Enter an email below to classify if it's spam or ham!")

    html_temp = """<div style = "background-color:#25246 ; padding:10px">
    <h2 style = "color:white; text-align:center;"> Spam Email Classification </h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html= True) # To render the html code as html

  # Getting the input from the user
    input_text = st.text_input(str("Enter the message"))

    spam_html = """
    <div style = background-color:#F4D03F; padding:10px >
    <h2 style = "color:white; text-align:center;"> This Email is Spam </h2>
    </div>
    """
    ham_html = """
    <div style = background-color:#F4D03F; padding:10px >
    <h2 style = "color:white; text-align:center;"> This Email is Ham </h2>
    </div>
    """

    if st.button("Click to predict"):
        output = predict_emails(input_text)
        


    st.success(output)


if __name__=='__main__':
    main()


# In[ ]: