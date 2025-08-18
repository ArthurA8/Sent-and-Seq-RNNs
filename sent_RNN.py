import torch 
import sklearn 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import torch.nn as nn 
import math 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from torch.utils.data import Dataset, DataLoader 
from torch.nn.utils.rnn import pad_sequence
from torch import optim 
from torch.nn import functional as F 
le = LabelEncoder()


"""Loading and Splitting"""

url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
df = pd.read_csv(url, delimiter='\t', header=None, names=['label', 'text'])

print(f'raw data: \n\n{df.head()}')
print(f'\nraw data shape: \n{df.shape}\n')
print("*" * 80 + "\n")

def text_preprop(text): 
    return text.lower().split() 

df['text'] = df['text'].apply(text_preprop)
df['label'] = le.fit_transform(df['label'])

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

train_text = train_data['text']
train_label = train_data['label']

test_text = test_data['text']
test_label = test_data['label']


print(f'cleaned train text: \n\n{train_data['text'].head()}')
print("*" * 80 + "\n")


"""Creating vocab"""

def vocab_func(data): 
    wrds = []
    for sentence in data: 
        for wrd in sentence: 
            wrds.append(wrd) 
    vocab = set(wrds)

    vocab_dict = {}
    n = 1 
    for i in vocab: 
        vocab_dict[i] = n 
        n += 1
    
    return(vocab_dict)

index_dict = vocab_func(df['text'])


"""Data indexing"""

def encode_phr(phrase): 
            return [index_dict[word] for word in phrase]

train_data['text'] = train_data['text'].apply(encode_phr)
test_data['text'] = test_data['text'].apply(encode_phr)

print(f'Indexed train text: \n\n{train_data['text'].head()}')
print("*" * 80 + "\n")


"""Dataset Class"""

class sentimentDataset(Dataset):
    def __init__(self, data):
        self.texts = data['text'].values
        self.labels = data['label'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

"""Padding using collate_fn"""

def collate_batch(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return texts_padded, labels


"""Fetching Dataset and Dataloader"""

batch_size = 32

train_dataset = sentimentDataset(train_data)
test_dataset= sentimentDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)


"""Network Architecture"""

class SentimentRNN(nn.Module): 
    def __init__(self, vocab_size, embed_dim, hidden_size, output_size):
        super(SentimentRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.fl = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x): 

        embedded_text = self.embedding(x)
        
        bs = x.size(0)
        h0 = torch.zeros(1, bs, self.hidden_size).to(x.device)

        out, hidden = self.rnn(embedded_text, h0)
        #print(hidden.shape)
        hidden_last = hidden[-1]
        logits = self.fl(hidden_last)
        #print(f'Network logits: {logits}')
        return logits


"""Initialize Network"""

vocab_size = len(index_dict) + 1
embed_dim = 128 
hidden_size = 128 
output_size = 2  #output size is 2 because there are two possible classes

network = SentimentRNN(vocab_size, embed_dim, hidden_size, output_size) 


"""Training"""

criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.Adam(network.parameters(), lr=0.0001)

num_epochs=200

epoch_losses = []
epoch_nums = []

for epoch in range(num_epochs):
    network.train() 
    epoch_loss=0

    batch_losses = []
    batch_nums = []

    batch_num = 0

    for preprop_txt, labels in train_loader: 

        batch_num += 1
        logits = network(preprop_txt)
        #print(logits)
        #print(len(logits))
        #print(logits.shape)
        #print(f'Label shape: {labels.shape, labels.dtype}')
        #print(torch.unique(labels))
        loss = criterion(logits, labels)

        optimizer.zero_grad() 
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step() 

        batch_losses.append(loss.item())
        batch_nums.append(batch_num)

        epoch_loss += loss.item()
    

    plt.plot(batch_nums, batch_losses)
    plt.xlabel("Batch Number")
    plt.ylabel("Batch Loss")
    #plt.show()
    plt.clf()

    print("EPOCH {}, LOSS {}".format(epoch, epoch_loss / len(train_loader)))

    avr_loss = epoch_loss / len(train_loader)
    epoch_losses.append(avr_loss)
    epoch_nums.append(epoch)


plt.plot(epoch_nums, epoch_losses) 
plt.xlabel("Epoch Number")
plt.ylabel("Epoch Loss")
plt.show()


"""Measuring Accuracy"""

def accuracy(model, test_loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad(): 

        batch_acc_arr = []
        num_batches = 0

        for preprop_txt, labels in test_loader: 

            num_batches += 1
            output = network(preprop_txt)
            #print(output)
            print(f'True batch classes: {labels}\n')
            pred_class = torch.argmax(output, dim=1)
            print(f'Pred batch classes: {pred_class}')
            print(f'\n')

            n = 0
            score = 0

            for i in range(len(preprop_txt)):
                if pred_class[n] == labels[n]: 
                    score += 1 
                n += 1

            batch_acc = (score / len(preprop_txt)) * 100 
            batch_acc_arr.append(batch_acc)
        
        accuracy = round(sum(batch_acc_arr) / num_batches, 2)
        print(f'Model accuracy: {accuracy}%')
        print("*" * 80 + "\n")

            
    
accuracy(network, test_loader)






    







    







 