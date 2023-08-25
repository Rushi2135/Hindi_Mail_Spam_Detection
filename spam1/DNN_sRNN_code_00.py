#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


# In[32]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


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
              metrics=['f1_score'])


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

params = {'legend.fontsize' : 'large',
          'figure.figsize'  : (12,9),
          'axes.labelsize'  : 'x-large',
          'axes.titlesize'  :'x-large',
          'xtick.labelsize' :'large',
          'ytick.labelsize' :'large',
         }
CMAP = plt.cm.coolwarm

plt.rcParams.update(params)

###-----------------------------------
### Function to plot Loss Curve
###-----------------------------------

def plot_tf_hist(hist_df):
    '''
    Args:
      hist_df : pandas Dataframe with four columns
                For 'x' values, we will use index
    '''
    fig, axes = plt.subplots(1,2 , figsize = (15,6))

    # properties  matplotlib.patch.Patch 
    props = dict(boxstyle='round', facecolor='aqua', alpha=0.4)
    facecolor = 'cyan'
    fontsize=12
    
    # Get columns by index to eliminate any column naming error
    y1 = hist_df.columns[0]
    y2 = hist_df.columns[1]
    y3 = hist_df.columns[2]
    y4 = hist_df.columns[3]

    # Where was min loss
    best = hist_df[hist_df[y3] == hist_df[y3].min()]
    
    ax = axes[0]

    hist_df.plot(y = [y1,y3], ax = ax, colormap=CMAP)


    # little beautification
    txtFmt = "{}: \n  train: {:6.4f}\n   test: {:6.4f}"
    txtstr = txtFmt.format(y1.capitalize(),
                           hist_df.iloc[-1][y1],
                           hist_df.iloc[-1][y3]) #text to plot
    
    # place a text box in upper middle in axes coords
    ax.text(0.3, 0.95, txtstr, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment='top', bbox=props)

    # Mark arrow at lowest
    ax.annotate(f'Min: {best[y3].to_numpy()[0]:6.4f}', # text to print
                xy=(best.index.to_numpy(), best[y3].to_numpy()[0]), # Arrow start
                xytext=(best.index.to_numpy()-1, best[y3].to_numpy()[0]), # location of text 
                fontsize=fontsize, va='bottom', ha='right',bbox=props, # beautification of text
                arrowprops=dict(facecolor=facecolor, shrink=0.05)) # arrow

    # Draw vertical line at best value
    ax.axvline(x = best.index.to_numpy(), color = 'green', linestyle='-.', lw = 3);

    ax.set_xlabel("Epochs")
    ax.set_ylabel(y1.capitalize())
    ax.set_title('Errors')
    ax.grid();
    ax.legend(loc = 'upper right') # model legend to upper left

    ax = axes[1]

    hist_df.plot( y = [y2, y4], ax = ax, colormap=CMAP)
    
    # little beautification
    txtFmt = "{}: \n  train: {:6.4f}\n  test:  {:6.4f}"
    txtstr = txtFmt.format(y2.capitalize(),
                           hist_df.iloc[-1][y2],
                           hist_df.iloc[-1][y4]) #text to plot

    # place a text box in upper middle in axes coords
    ax.text(0.3, 0.2, txtstr, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment='top', bbox=props)

    # Mark arrow at lowest
    ax.annotate(f'Best: {best[y4].to_numpy()[0]:6.4f}', # text to print
                xy=(best.index.to_numpy(), best[y4].to_numpy()[0]), # Arrow start
                xytext=(best.index.to_numpy()-1, best[y4].to_numpy()[0]), # location of text 
                fontsize=fontsize, va='bottom', ha='right',bbox=props, # beautification of text
                arrowprops=dict(facecolor=facecolor, shrink=0.05)) # arrow
    
    
    # Draw vertical line at best value
    ax.axvline(x = best.index.to_numpy(), color = 'green', linestyle='-.', lw = 3);

    ax.set_xlabel("Epochs")
    ax.set_ylabel(y2.capitalize())
    ax.grid()
    ax.legend(loc = 'lower right')
    
    plt.tight_layout()


# In[48]:


plot_tf_hist(hist_df)


# In[ ]:





# In[ ]:




