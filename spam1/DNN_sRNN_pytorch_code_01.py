#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score


# In[26]:


import nltk
nltk.download('punkt')


# In[2]:


df = pd.read_csv("./final_data1.csv")
df


# In[ ]:


# checking for cuda
print ('Is CUDA available: ', torch.cuda.is_available())
print ('CUDA version: ', torch.version.cuda )
print ('Current Device ID: ', torch.cuda.current_device())
print ('Name of the CUDA device: ', torch.cuda.get_device_name(torch.cuda.current_device()))


# In[23]:


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# In[28]:


x , y = df['Hindi'] , df['label']


# In[29]:


lbl = LabelEncoder()
y = lbl.fit_transform(y)


# In[30]:


X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[31]:


X_train = X_train.to_numpy()
X_test = X_test.to_numpy()


# In[32]:


y_train , y_test = np.array(y_train), np.array(y_test)


# In[33]:


def preprocess_text(text):
    # You may need to implement more advanced preprocessing steps
    return text.split()

# Convert text data to numerical data
vocab = set()
for text in x:
    tokens = preprocess_text(text)
    vocab.update(tokens)


# In[34]:


vocab_size = len(vocab)
embedding_dim = 100  # you may need to change this according to your task 100 is standard value
max_length = max(len(seq) for seq in x)
trunc_type= 'post'


# In[35]:


# Tokenize and pad sequences
def tokenize_hindi(text):
    tokens = word_tokenize(text,language='hindi',preserve_line=True)
    return tokens

trn_seq = [tokenize_hindi(item) for item in X_train]
tst_seq = [tokenize_hindi(item) for item in X_test]


# In[36]:


word_to_idx = {}
for seq in trn_seq + tst_seq:
    for word in seq:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)


# In[37]:


trn_pad = [torch.tensor([word_to_idx[word] for word in seq]) for seq in trn_seq]
trn_pad = nn.utils.rnn.pad_sequence(trn_pad, batch_first=True, padding_value=0)
tst_pad = [torch.tensor([word_to_idx[word] for word in seq]) for seq in tst_seq]
tst_pad = nn.utils.rnn.pad_sequence(tst_pad, batch_first=True, padding_value=0)


# In[38]:


# Define the RNN Model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, 10),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        embedded = self.embedding(x)
        rnn_output, _ = self.rnn(embedded)
        rnn_output = rnn_output[:, -1, :]  # Get the last output of RNN sequence
        output = self.fc(rnn_output)
        return output


# In[39]:


# Instantiate the model
hidden_dim = 256
alpha = 0.001
model_rnn = RNNModel(vocab_size, embedding_dim, hidden_dim)

# Define loss function and optimizer
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model_rnn.parameters(),lr=alpha)


# In[41]:


# Create DataLoader
train_dataset = TensorDataset(trn_pad, torch.tensor(y_train, dtype=torch.float))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(tst_pad, torch.tensor(y_test, dtype=torch.float))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# In[1]:


# for df

loss, tloss = [],[]
acc, tacc = [],[]
n_epoch = []

# Training loop
for epoch in range(5):
    model_rnn.train()
    train_loss = 0.0
    train_acc = 0.0
    for batch_inputs, batch_labels in train_loader:
        optimizer.zero_grad()
        batch_outputs = model_rnn(batch_inputs.long())

        predicted = torch.round(batch_outputs)

        batch_loss = loss_fn(batch_outputs.squeeze(), batch_labels)
        batch_acc = accuracy_score(batch_labels.cpu().numpy(),
                                   predicted.detach().cpu().numpy())
        batch_loss.backward()
        optimizer.step()

        train_loss += batch_loss.item() * batch_inputs.size(0)
        train_acc += batch_acc * batch_inputs.size(0)

    train_loss /= len(train_dataset)
    train_acc /= len(train_dataset)

    loss.append(train_loss)
    acc.append(train_acc)

    # Evaluation
    model_rnn.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            batch_outputs = model_rnn(batch_inputs.long())
            predicted = torch.round(batch_outputs)

            batch_loss = loss_fn(batch_outputs.squeeze(), batch_labels)
            batch_acc = accuracy_score(batch_labels.cpu().numpy(),
                                       predicted.detach().cpu().numpy())

            test_loss += batch_loss.item() * batch_inputs.size(0)
            test_acc += batch_acc* batch_inputs.size(0)

            
        test_loss /= len(test_dataset)
        test_acc /= len(test_dataset)

        tloss.append(test_loss)
        tacc.append(test_acc)

    n_epoch.append(epoch)

    print(f'MINE: At epoch {epoch} | Loss - train:{train_loss:.4f}, test:{test_loss:.4f} | Acc - train:{train_acc:.4f}, test:{test_acc:.4f}')


# In[ ]:


loss_df = pd.DataFrame({'epoch' : n_epoch, 'loss' : loss, 'test_loss': tloss, 'acc' : acc, 'test_acc': tacc})
print(loss_df.head())


# In[ ]:


loss_df.to_csv("sRNN_code00.csv")


# In[ ]:


model_rnn.save("model_rnn_pytorch.h5")


# In[ ]:
