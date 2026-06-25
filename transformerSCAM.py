import torch 
import torch.nn as nn
import numpy as np
import csv
import re
import sys
import random
from datasets import load_dataset
import csv
import json
dataset = load_dataset("wangyuancheng/discord-phishing-scam-clean")
ds = load_dataset("wangyuancheng/discord-phishing-scam")
all_texts = [row['msg_content'] for row in dataset['train']]
all_labels = [float(row['label']) for row in dataset['train']]
with open("scmas.csv","r",newline='',encoding="utf-8-sig") as f:
    reader=csv.DictReader(f)
    for row in reader:
        all_texts.append(row.get("message"))
        all_labels.append(float(row.get("label")))
for rowd in ds['train']:
    all_texts.append(rowd['msg_content'])
    all_labels.append(float(rowd['lable']))
with open("scam2.csv","r",newline='',encoding="utf-8-sig") as f:
    reader=csv.DictReader(f)
    for row in reader:
        all_texts.append(row.get("message"))
        all_labels.append(float(row.get("label")))
with open("discord_scam_dataset_v2.csv","r",newline='',encoding="utf-8-sig") as f:
    reader=csv.DictReader(f)
    for row in reader:
        all_texts.append(row.get("message"))
        all_labels.append(float(row.get("label")))
with open("discord_scam_dataset3.csv","r",newline='',encoding="utf-8-sig") as f:
    reader=csv.DictReader(f)
    for row in reader:
        all_texts.append(row.get("message"))
        all_labels.append(float(row.get("label")))

first_Y=[]
vocab = {"<PAD>": 0, "<UNK>": 1}
first_X=[]
for row in all_texts:
    k=row.lower()
    clean_msg = re.sub(r'([?.!,;:])', r' \1 ', k) 
    clean_msg = re.sub(r'\s+', ' ', clean_msg)
    for word in  clean_msg.split() :
        if word not in vocab:
            vocab[word]=len(vocab)
first_X=all_texts
first_Y=all_labels
print(len(first_X),len(first_Y))
with open("vocab.json", "w") as f:
    json.dump(vocab, f)
print("saved")
def text_to_numbers(sentence,vocab ,max_lean=50):
        words=sentence.lower()
        clean_msg = re.sub(r'([?.!,;:])', r' \1 ', words)
        clean_msg = re.sub(r'\s+', ' ', clean_msg)
        words=clean_msg.split()
        sentence_ids=[]
        for word in words:
            word_id=vocab.get(word,1)
            sentence_ids.append(word_id)
        if len(sentence_ids) > max_lean:
            sentence_ids=sentence_ids[:50]
        else:
            how_short=max_lean-len(sentence_ids)
            sentence_ids=sentence_ids+[0]*how_short
        return sentence_ids
secand_X=[]
for message in first_X:
    secand_X.append(text_to_numbers(message,vocab))
combined = list(zip(secand_X, first_Y))
random.shuffle(combined)
secand_X, first_Y = zip(*combined)
split=int(len(secand_X) * 0.8)
secand_X_train=torch.tensor(secand_X[:split])
secand_X_test=torch.tensor(secand_X[split:])
secand_Y_train=torch.tensor(first_Y[:split]).unsqueeze(1)
secand_Y_test=torch.tensor(first_Y[split:]).unsqueeze(1)
full_mask = (torch.tensor(secand_X) == 0)
mask_train = full_mask[:split]
mask_test = full_mask[split:]
class encoder(nn.Module):
    def __init__(self,embed_size=128 ,num_heads=4,ff_size=256):
        super().__init__()
        self.attention=nn.MultiheadAttention(embed_size,num_heads,batch_first=True)
        self.agent=nn.Linear(embed_size,ff_size)
        self.boss=nn.Linear(ff_size,embed_size)
        self.norm1=nn.LayerNorm(embed_size)
        self.norm2=nn.LayerNorm(embed_size)
        self.dropout=nn.Dropout(0.3)
    def forward(self,x,mask=None):
        att_output, _=self.attention(x,x,x,key_padding_mask=mask)
        x=self.norm1(x+att_output)
        liner=self.dropout(torch.nn.functional.gelu(self.agent(x)))
        liner=self.boss(liner)
        x=self.norm2(x+liner)
        return x
class ScamBrain(nn.Module):
    def __init__(self, vocab_size,embed_size):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.gps=nn.Parameter(torch.randn(1,50,embed_size))
        self.encoder=encoder(embed_size)
        self.classifier = nn.Linear(embed_size, 1)
    def forward(self , x,mask=None):
        x=self.embedding(x)
        x=x+self.gps
        x=self.encoder(x,mask=mask)
        x=x.mean(dim=1)
        return self.classifier(x)
model =ScamBrain(vocab_size=len(vocab),embed_size=128)
edt=torch.optim.Adam(model.parameters(),lr=0.0001)
creation=nn.BCEWithLogitsLoss()
for epoch in range(175):
    edt.zero_grad()
    outpute=model(secand_X_train,mask=mask_train)
    loss=creation(outpute , secand_Y_train)
    loss.backward()
    edt.step()
    if (epoch + 1) %25 == 0:
        print(f"Epoch {epoch+1}/120 | Loss: {loss.item():.4f}")
print(f"last loss{loss}")
model.eval()
with torch.no_grad():
    outpute=model(secand_X_test,mask=mask_test)
    loss = creation(outpute, secand_Y_test)
    predicted = (torch.sigmoid(outpute) > 0.5).float()
    accuracy = (predicted == secand_Y_test).float().mean()
    print(f"Loss: {loss.item():.4f}")
    print(f"Accuracy: {accuracy.item() * 100:.2f}%")
torch.save(model.state_dict(),'transBrain.pth')