#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import streamlit as st
import torch
from torch.utils.data import DataLoader ,Dataset
from transformers import AutoTokenizer,BertForQuestionAnswering,AutoModel


# In[3]:


from transformers import AutoTokenizer,BertForQuestionAnswering,AutoModel
model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# In[4]:


from transformers import DataCollatorWithPadding


# In[5]:


torch.set_default_device('cpu')


# In[6]:


from transformers import BertTokenizer, BertModel


# In[7]:


class bert_compare(torch.nn.Module):
    def __init__ (self):
        super(bert_compare,self).__init__()
        self.bert=BertModel.from_pretrained("bert-base-uncased")
        
        self.Linear=torch.nn.Linear(768,30 )
        self.elu=torch.nn.ELU()
        self.Linear2=torch.nn.Linear(280 ,1 )
        self.cnn1=torch.nn.Conv1d(768,256,kernel_size=2)
        self.cnn2=torch.nn.Conv1d(256,10,kernel_size=2)

        self.relu=torch.nn.ReLU()
    def forward(self,x):
        x=self.bert(**x).last_hidden_state
        x=x.permute(0,2,1)
        x=self.cnn1(x)
        x=self.relu(x)
        x=self.cnn2(x)
        x=torch.nn.Flatten()(x)
        x=self.Linear2(x)
        return x


# In[8]:


model=bert_compare()
optim=torch.optim.AdamW(model.parameters(),lr=5e-5)
loss=torch.nn.BCEWithLogitsLoss()


# In[9]:


def tok(x,y):
        out=tokenizer(x,y, truncation=True, max_length=30,padding='max_length', return_tensors="pt")
        out={key:value for key,value in out.items()}
        return out
h=tok('my name is mohamed','what is your name')    
model(h)


# In[10]:


model=torch.load('Downloads/model9.pth',map_location=torch.device('cpu'))


# In[11]:


word=['my name is mohamed ', "How do I read and find my YouTube comments?" ,"How can I see all my Youtube comments?","How can Internet speed be increased by hacking through DNS?","What is the step by step guide to invest in share market in india?","where is capital of egypt?",'when did you born ','what is your name',"what is capital of egypt",'how old are you']


# In[19]:


import gradio as gr


# In[12]:


def tok(x,y):
        out=tokenizer(x,y, truncation=True, max_length=30,padding='max_length', return_tensors="pt")
        out={key:value for key,value in out.items()}
        return out
for i in range(9):
        r=torch.randint(len(word),size=(1,))
        r2=torch.randint(len(word),size=(1,))
        h=tok(word[r],word[r2])    
        e=model(h)
        ans= 'the same' if  int(torch.sigmoid( e)>=.5) else 'not the same'
        print (f'{word[r]} is {ans} {word[r2]}'  )


# In[32]:


def sentance_calcute(sentance1,sentance2) ->(int,str) :    
        out=tokenizer(sentance1,sentance2, truncation=True, max_length=30,padding='max_length', return_tensors="pt")
        h={key:value for key,value in out.items()}
        e=model(h)
        ans=torch.sigmoid( e)
        ans2='Same' if ans>=.5 else 'Not same'
        return ans,ans2


# In[46]:


input_color = "lightred"  # Change the color of the input fields

iface = gr.Interface(
    fn=sentance_calcute,
    inputs=["text", "text"],
    outputs=["number", "text"],
    layout="horizontal",
    title="Sentence Similarity Checker",
    description="Enter two sentences to check their similarity.",
    examples=[
        ["The sun is in the west.", "The sun goes down in the west."],
        ["Why is biodiversity important for ecosystems?", "She is extremely joyful."],
        ["The cat is sleeping on the chair.", "The cat is napping on the chair."]
       ,["Why is biodiversity important for ecosystems?", "When did the Renaissance period begin?"]
    ],

)

# Launch the interface
iface.launch()


# In[ ]:




