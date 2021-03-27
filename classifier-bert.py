#!/usr/bin/env python3
# coding: utf-8

# Copyright (C) 2021  Serge Sharoff
# A multi-label genre classifier for tje Huggingface transformers module

import time
starttime=int(time.time())

import sys
import json
import argparse

import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

from collections import defaultdict

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader,RandomSampler
import torch.nn.functional as F
import transformers

parser = argparse.ArgumentParser(description="A Transformer Model for Genre Classification")
parser.add_argument('--mname', type=str, default='bert-base-multilingual-cased', help='Model type')
parser.add_argument('--tname', type=str, default='', help='Tokenizer')
parser.add_argument('--testmodel', type=str, help='Save model file in H5')
parser.add_argument('-p', '--prefix', type=str, default='', help='defaults for input files, annotations and dictionaries')
parser.add_argument('-i', '--inputfile', type=str, default='.ol', help='one-doc-per-line training corpus')
parser.add_argument('-a', '--annotations', type=str, default='.csv', help='tab-separated FTDs with the header and text ids')
parser.add_argument('-b', '--binary', type=float, default=0, help='to treat annotations as binary above a threshold')
parser.add_argument('-t', '--testfile', type=str, help='one-doc-per-line test only corpus')

parser.add_argument('-x', '--maxlen', type=int, default=400, help='to shorten docs')
parser.add_argument('-g', '--gensplit', type=int, default=0, help='to generate extra examples if longer than maxlen')

parser.add_argument('-l', '--loss', type=str, default='binary_crossentropy', help='loss for training')
parser.add_argument('-m', '--metrics', type=str, default='mae', help='metrics for validation')
parser.add_argument('-s', '--seed', type=int, default=42)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('-e', '--epochs', type=int, default=4)
parser.add_argument( '--batch_size', type=int, default=64)

parser.add_argument( '--valsplit', type=float, default=0.05)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('-c', '--cv_folds', type=int, default=0)
parser.add_argument('-k', '--topk', type=int, default=2, help='topK predicted labels to output')
parser.add_argument('-v', '--verbosity', type=int, default=1)

outname=re.sub(' ','=','_'.join(sys.argv[min(len(sys.argv),2):]))  
if args.verbosity>0:
    print('Parameter list: %s' % outname, file=sys.stderr)

args = parser.parse_args()

if args.seed:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
print(device)

if len(args.prefix)>0:
    args.inputfile=args.prefix+args.inputfile
    args.annotations=args.prefix+args.annotations

y_train = pd.read_csv(args.annotations,header=0,index_col=0,sep='\t')
if args.binary>0:
    binfunc=lambda x : 1 if x>args.binary else 0
    y_train = y_train.applymap(binfunc)
else:
    maxval=y_train.max()
    y_train = y_train / maxval  #[0..2] annotations need to be within 

if args.verbosity>0:
    print('Train data: %d train, %d labels' % (len(X_train), len(y_train)), file=sys.stderr)



class_names = ['Simple' ,'Moderate','Complex']
df = pd.read_tsv("Albert_All.csv")
print(df.head())

EPOCHS = int(sys.argv[1]) if len(sys.argv)>1 else 4


tokenizer = transformers.AutoTokenizer.from_pretrained(args.mname)

MAX_LEN = 280


# In[26]:


# a hack from the sentiment classifier
class GPReviewDataset(Dataset):

  def __init__(self, reviews, targets, tokenizer, max_len):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.reviews)
  
  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      padding="max_length",
      return_attention_mask=True,truncation=True,
      return_tensors='pt',
    )

    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }


###Data_Loader Helper
def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = GPReviewDataset(
    reviews=df.Text.to_numpy(),
    targets=df.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )


###Train_Validation_Test split
df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)


print(df_train.shape, df_val.shape, df_test.shape)


df_train_Sample1,df_rest= train_test_split(df_train, test_size=0.9, random_state=RANDOM_SEED)
df_train_Sample2,df_rest= train_test_split(df_train, test_size=0.8, random_state=RANDOM_SEED)
df_train_Sample3,df_rest= train_test_split(df_train, test_size=0.7, random_state=RANDOM_SEED)
df_train_Sample4,df_rest= train_test_split(df_train, test_size=0.6, random_state=RANDOM_SEED)
df_train_Sample5,df_rest= train_test_split(df_train, test_size=0.5, random_state=RANDOM_SEED)
df_train_Sample6,df_rest= train_test_split(df_train, test_size=0.4, random_state=RANDOM_SEED)
df_train_Sample7,df_rest= train_test_split(df_train, test_size=0.3, random_state=RANDOM_SEED)
df_train_Sample8,df_rest= train_test_split(df_train, test_size=0.2, random_state=RANDOM_SEED)
df_train_Sample9,df_rest= train_test_split(df_train, test_size=0.1, random_state=RANDOM_SEED)



print(len(df_train_Sample1),len(df_train_Sample2),len(df_train_Sample3),len(df_train_Sample4),len(df_train_Sample5),len(df_train_Sample6),len(df_train_Sample7),len(df_train_Sample8),len(df_train_Sample9))

BATCH_SIZE = 16

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)


train_data_loader_sample1 = create_data_loader(df_train_Sample1, tokenizer, MAX_LEN, BATCH_SIZE)
train_data_loader_sample2 = create_data_loader(df_train_Sample2, tokenizer, MAX_LEN, BATCH_SIZE)
train_data_loader_sample3 = create_data_loader(df_train_Sample3, tokenizer, MAX_LEN, BATCH_SIZE)
train_data_loader_sample4 = create_data_loader(df_train_Sample4, tokenizer, MAX_LEN, BATCH_SIZE)
train_data_loader_sample5 = create_data_loader(df_train_Sample5, tokenizer, MAX_LEN, BATCH_SIZE)
train_data_loader_sample6 = create_data_loader(df_train_Sample6, tokenizer, MAX_LEN, BATCH_SIZE)
train_data_loader_sample7 = create_data_loader(df_train_Sample7, tokenizer, MAX_LEN, BATCH_SIZE)
train_data_loader_sample8 = create_data_loader(df_train_Sample8, tokenizer, MAX_LEN, BATCH_SIZE)
train_data_loader_sample9 = create_data_loader(df_train_Sample9, tokenizer, MAX_LEN, BATCH_SIZE)


##Creatinng list of the training iteration
train_loader = []
train_loader.append(train_data_loader_sample1)
train_loader.append(train_data_loader_sample2)
train_loader.append(train_data_loader_sample3)
train_loader.append(train_data_loader_sample4)
train_loader.append(train_data_loader_sample5)
train_loader.append(train_data_loader_sample6)
train_loader.append(train_data_loader_sample7)
train_loader.append(train_data_loader_sample8)
train_loader.append(train_data_loader_sample9)
train_loader.append(train_data_loader)

class ReadabilityClassifier(nn.Module):
  def __init__(self, n_classes):
    super(ReadabilityClassifier, self).__init__()
    self.bert = transformers.AutoModel.from_pretrained(args.mname)  # ForSequenceClassification
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    X= self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    _=X[0]
    pooled_output=X[1] 
    output = self.drop(pooled_output)
    return self.out(torch.as_tensor(output))

##Set the model to GPU
model = ReadabilityClassifier(len(class_names))
model = model.to(device)


# ### Training

##parameters, Epochs, optimizer, Learning rate
optimizer = transformers.AdamW(model.parameters(), lr=args.lr)  #, correct_bias=False
total_steps = len(train_data_loader) * args.epochs

scheduler = transformers.get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=100,
  num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)



def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0
  
  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)

history = defaultdict(list)
best_accuracy = 0
i=1
for sample in train_loader:
  print(f'Sample {i} / {len(train_loader)}')
  print('-' * 10)
  i=i+1
  for epoch in range(EPOCHS):
     
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    
    train_acc, train_loss = train_epoch(
        model,
        sample,    
        loss_fn, 
        optimizer, 
        device, 
        scheduler, 
        len(df_train)
        )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn, 
        device, 
        len(df_val)
        )

    print(f'Val   loss {val_loss} accuracy {val_acc}')

    test_acc, test_loss = eval_model(
        model,
        test_data_loader,
        loss_fn, 
        device, 
        len(df_test)
        )

    print(f'test   loss {test_loss} accuracy {test_acc}')

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    history['test_acc'].append(test_acc)
    history['test_loss'].append(test_loss)
    
    if val_acc > best_accuracy:
      torch.save(model.state_dict(), f'best_model_{args.mname}.bin')


test_acc, _ = eval_model(
  model,
  test_data_loader,
  loss_fn,
  device,
  len(df_test)
)

test_acc.item()



def get_predictions(model, data_loader):
  model = model.eval()
  
  review_texts = []
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():
    for d in data_loader:

      texts = d["review_text"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      probs = F.softmax(outputs, dim=1)

      review_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(probs)
      real_values.extend(targets)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return review_texts, predictions, prediction_probs, real_values


# # In[48]:


# ##New Prediction Data
# df_test=pd.read_excel('/content/Eval_New_Saqq.xlsx')
# test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)


# print(df_test)


# y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
#   model,
#   test_data_loader
# )


# # In[54]:


# print(classification_report(y_test, y_pred, target_names=['A','C']))


# # In[57]:


# def show_confusion_matrix(confusion_matrix):
#   hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
#   hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
#   hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
#   plt.ylabel('True level')
#   plt.xlabel('Predicted difficulty');

# cm = confusion_matrix(y_test, y_pred)
# df_cm = pd.DataFrame(cm, index=['A','C'], columns=['A','C'])
# show_confusion_matrix(df_cm)


# In[ ]:




