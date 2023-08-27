#!/usr/bin/env python
# coding: utf-8

# In[1]:

# pip install scikit-learn
# !pip install tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

# In[2]:


df = pd.read_csv("./final_data1.csv")
df


# In[21]:


x , y = df['Hindi'] , df['label']


# In[23]:


lbl = LabelEncoder()
y = lbl.fit_transform(y)


# In[24]:


X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[26]:


X_train = X_train.to_numpy()
X_test = X_test.to_numpy()


# In[27]:


y_train , y_test = np.array(y_train), np.array(y_test)


# In[28]:


def preprocess_text(text):
    # You may need to implement more advanced preprocessing steps
    return text.split()

# Convert text data to numerical data
vocab = set()
for text in x:
    tokens = preprocess_text(text)
    vocab.update(tokens)


# In[29]:


vocab_size = len(vocab)
embedding_dim = 100  # you may need to change this according to your task 100 is standard value
max_length = max(len(seq) for seq in x)
trunc_type= 'post'

# In[164]:

# Specify GPU device
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# In[32]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[33]:


tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)
word_idx = tokenizer.word_index
len(word_idx)


# In[34]:


trn_seq = tokenizer.texts_to_sequences(X_train)
trn_pad = pad_sequences(trn_seq,maxlen=max_length,truncating=trunc_type)

tst_seq = tokenizer.texts_to_sequences(X_test)
tst_pad = pad_sequences(tst_seq,maxlen=max_length)


# ### Using RNN

# In[164]:


from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Embedding, Dropout


# In[165]:


model_rnn = Sequential([
    Embedding(vocab_size, embedding_dim,
                             input_length=max_length),
    SimpleRNN(256),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.25),
    Dense(10, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model_rnn.summary()


# In[166]:


model_rnn.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[167]:


history = model_rnn.fit(trn_pad, y_train, epochs=50,
                  validation_data = (tst_pad, y_test),verbose=1)


# In[41]:


hist_df = pd.DataFrame(history.history)
print(hist_df.head())


# In[54]:


print(hist_df.describe())


# In[ ]:


###------------------
### parameters to plot Loss curve
###------------------

# In[48]:


model_rnn.save("my_rnn_model.h5")
hist_df.to_csv("rnn_loss_df.csv")
# In[ ]:





# In[ ]:




