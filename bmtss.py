#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install transformers')


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


get_ipython().system(' pip install tqdm')


# In[11]:


from  tqdm import tqdm


# In[12]:


model.train()


# In[13]:


model.bert.train()


# In[14]:


model=torch.load('Downloads/model9.pth',map_location=torch.device('cpu'))


# In[15]:


word=['my name is mohamed ', "How do I read and find my YouTube comments?" ,"How can I see all my Youtube comments?","How can Internet speed be increased by hacking through DNS?","What is the step by step guide to invest in share market in india?","where is capital of egypt?",'when did you born ','what is your name',"what is capital of egypt",'how old are you']


# In[16]:


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


# In[17]:



        h=tok("what is capital of egypt","when is  capital of egypt")    
        e=model(h)
        ans= 'the same' if  int(torch.sigmoid( e)>=.5) else 'not the same'
        print (f' {ans} '  )


# In[19]:


def are_sentences_same(sentence1, sentence2):
    doc1=tok(sentence1,sentence2)   
    out_model=model(doc2)
    ans= 'the same' if  int(torch.sigmoid( out_model)>=.5) else 'not the same'

    return torch.sigmoid( ans)

def main():
    st.title('Sentence Similarity Checker')
    st.write('Enter two sentences to check if they are the same.')

    # Input sentences
    sentence1 = st.text_input('Enter the first sentence:')
    sentence2 = st.text_input('Enter the second sentence:')

    # Check if both sentences are provided
    if sentence1 and sentence2:
        similarity_score = are_sentences_same(sentence1, sentence2)
        st.write(f'Similarity Score: {similarity_score:.2f}')

        if similarity_score >= 0.5:
            st.write('The sentences are very similar.')
        else:
            st.write('The sentences are different.')

if __name__ == '__main__':
    main()


# In[ ]:




