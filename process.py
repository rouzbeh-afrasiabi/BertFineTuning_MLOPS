from BertFineTuning.utils import *
from BertFineTuning.data_config import *

import importlib
import os
import sys
import math
import copy
import urllib
import zipfile
import requests
import shutil
import re
from collections import OrderedDict
import warnings

from azureml.core import Run
from azureml.core import Workspace, Datastore
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.datastore import Datastore
from azureml.data.data_reference import DataReference

pip_packages=[
              "azureml-sdk==1.0.17", "scikit-learn==0.21.3",
              "download==0.3.4", "pandas==0.25.1",
              "spacy==2.1.4", "numpy==1.17.2","pytorch_transformers==1.0.0",
              ]

for item in pip_packages:
  try:
    importlib.import_module(item.split("=")[0])
  except:
    _command="pip install --upgrade "+item
    os.system(_command)

from sklearn.utils.class_weight import compute_class_weight
import spacy
from download import download
import numpy as np
import pandas as pd


cwd = str(os.getcwd())
sys.path.append(cwd)
sys.path.insert(0, cwd)

def download_files(files,download_folder):
    for file in files:
        [[_,location]]=file.items()
        file_name=os.path.basename(location)
        exists,_=check_file(file_name,download_folder)
        if(exists):
            pass
        else:
            print('*** Downloading : ',file_name)
            try:
                r = requests.get(location, auth=('usrname', 'password'), verify=False,stream=True)
                r.raw.decode_content = True   
                with open(os.path.join(download_folder,file_name), 'wb') as f:
                        shutil.copyfileobj(r.raw, f)
            except:
                raise Exception('Failed')

def create_folders(folders):
    for folder in folders:
        if(check_folder(folder)):
            pass
        else:
            os.mkdir(folder)
            
def get_args():

  parser = argparse.ArgumentParser()
  parser.add_argument("--processed_data_ref",  type=str, dest="processed_data_ref")
  parser.add_argument("--input_data_ref",  type=str, dest="input_data_ref")
  parser.add_argument("--tenant_id",  type=str, dest="tenant_id")
  parser.add_argument("--application_id",  type=str, dest="application_id")
  parser.add_argument("--app_secret",  type=str, dest="app_secret")
  parser.add_argument("--subscription_id",  type=str, dest="subscription_id")
  parser.add_argument("--resource_group",  type=str, dest="resource_group")
  parser.add_argument("--workspace_name",  type=str, dest="workspace_name")
  parser.add_argument("--workspace_region",  type=str, dest="workspace_region")
  parser.add_argument("--object_id",  type=str, dest="object_id") 
  args = parser.parse_args()

  return(args)

def get_ws(args):
  
  tenant_id = args.tenant_id
  application_id = args.application_id
  app_secret = args.app_secret
  subscription_id = args.subscription_id
  resource_group = args.resource_group
  workspace_name = args.workspace_name
  workspace_region = args.workspace_region
  object_id = args.object_id
  
  service_principal = ServicePrincipalAuthentication(
          tenant_id=tenant_id,
          service_principal_id=application_id,
          service_principal_password=app_secret)

  ws = Workspace.get(
              name=workspace_name,
              subscription_id=subscription_id,
              resource_group=resource_group,
              auth=service_principal)
  return(ws)

def is_blob(target_blob_store='',path=''):    
    blob_container_name=target_blob_store.container_name
    return(target_blob_store.blob_service.exists(blob_container_name,path))
  
  
    
def remove_stop_words(doc,nlp):
    output_string=[]
    for token in doc:
        if (not token.is_stop):
            output_string.append(token.text)
        else:
            pass
    new_doc=nlp(" ".join(output_string))
    return(doc)

def fix_oov(doc,nlp):
    #spell check was removed because it affects the results
#     from spellchecker import SpellChecker
    output=[]
#     spell = SpellChecker()
    for token in doc:
        new_token=''
        if(token.text in nlp.vocab):
            new_token=token.text
#         elif(token.text!=spell.correction(token.text)):
#             new_token=spell.correction(token.text)
#             print(token.text,' spelling changed to: ',new_token)
        else:
          new_token=replace_punct(token.text)
#           print(token.text,"not found in vocabulary, changed to: ",new_token)
        if(new_token):
            output.append(new_token)
    new_doc=nlp(" ".join(output))
    return(new_doc)
  
def lemmatize(doc,nlp):
    output_string=[]
    for token in doc:
        output_string.append(token.lemma_)
    new_doc=nlp(" ".join(output_string))
    return(new_doc)

def remove_punct(doc,nlp):
    output_string=[]
    for token in doc:
        if((not token.is_punct) and (not token.is_space)):
            output_string.append(token.lemma_)
    new_doc=nlp(" ".join(output_string))
    return(new_doc)


def to_index(doc):
    ids=[]
    for token in doc:
        if token.has_vector:
            id = nlp.vocab.vectors.key2row[token.norm]
        else:
            id = None 
        ids.append(id)
    return(ids)

def replace_punct(input_string):
    import string
    output=input_string.translate(str.maketrans(string.punctuation,
                 ' ' * len(string.punctuation))).replace(' '*4, ' ').replace(' '*3, ' ').replace(' '*2, ' ').strip()
    
    return(output)
def remove_numbers(doc,nlp):
    output_string=[]
    for token in doc:
        if(not token.is_digit):
            output_string.append(token.lemma_)
    new_doc=nlp(" ".join(output_string))
    return(new_doc)  

def to_vec(doc,nlp):
    vecs=[]
    for token in doc:
        if token.has_vector:
            vec = token.vector
        else:
            vec = None 
            print('empty vector generated!!')
        vecs.append(vec)
    return(np.array(vecs))

def to_piece(input):
    global nlp
    doc=nlp(input)
    global sp
    pieces=sp.EncodeAsPieces(doc.text)
    if(pieces):
        return(pieces)
    else:
        return(sp.EncodeAsPieces("<unk>"))
    
def process_doc(input_string):
    global _processed
    global nlp
    global _max_seq_len
    doc=nlp(str(input_string).lower())
    doc_oov=fix_oov(doc,nlp)
    doc_lemma=lemmatize(doc_oov,nlp)
    doc_stop=remove_stop_words(doc_lemma,nlp)
    doc_punct=remove_punct(doc_stop,nlp)
    if((_processed+1)%100000==0): 
        print('processed ',_processed+1, 'records')
    _processed=_processed+1

    return(doc_punct)
  
def split_dataset(target_df, fracs, weights):
  global random_state
  datasets = []
  in_features = target_df.copy()
  for frac in fracs:
      temp = in_features.sample(frac=frac, replace=False,
                                random_state=random_state,
                                weights=weights)
      in_features.drop(temp.index, inplace=True)
      datasets.append(temp)
  return [data for data in datasets]
  
  
if __name__ == '__main__':
    random_state=123
    global run
    run = Run.get_context()
    _params=get_args()
    ws=get_ws(_params)
    
    processed_data_ref=_params.processed_data_ref
    input_data_ref=_params.input_data_ref
    
    def_blob_store = Datastore(ws, 'workspaceblobstore')
    blob_container_name=def_blob_store.container_name
        

    
    if(not is_blob(def_blob_store,'data/cleaned/Main.csv')):
      print('Processing input data ...')
      try:
          import en_vectors_web_lg
          nlp = en_vectors_web_lg.load()
      except:
          os.system("pip install "+ '{}/install/en_vectors_web_lg-2.1.0.tar.gz'.format(input_data_ref))
          import en_vectors_web_lg
          nlp = en_vectors_web_lg.load() 
          
      _processed=0
      train_df=pd.read_csv('{}/original/train.csv'.format(input_data_ref),encoding='utf-8',sep=',', engine='python')
      question1_clean=train_df.question1.apply(lambda x:process_doc(x))
      question2_clean=train_df.question2.apply(lambda x:process_doc(x))
      main_df=train_df.copy()
      main_df['question2']=question2_clean
      main_df['question1']=question1_clean
      main_df.to_csv('Main.csv')
      #This is the best method of creating a blob, upload will fail
      def_blob_store.blob_service.create_blob_from_path(container_name=blob_container_name,
                                                        blob_name='data/cleaned/Main.csv',
                                                        file_path='Main.csv')
    else:
      print('Main.csv exists, loading old processed file!!')
      train_df=pd.read_csv('{}/cleaned/Main.csv'.format(input_data_ref), index_col=[0])

    train_df=train_df.dropna(axis=0,how='any')
    weights=compute_class_weight('balanced',train_df['is_duplicate'].unique(),train_df['is_duplicate'].values.flatten())
    weights_all=train_df['is_duplicate'].map(dict(zip(train_df['is_duplicate'].unique(),weights)))
    
    target_dataset = train_df
    names = ['train_split', 'valid_split', 'test_split']
    
    if not all([is_blob(def_blob_store,'data/cleaned/'+name+'.csv') for name in names]):
        split_data = split_dataset(target_dataset, [0.6, .5, 1],
                                   weights_all)
        split_data_dict = dict(zip(names, [data for data in split_data]))
        for (i, data) in enumerate(split_data):
            data.to_csv(names[i]+'.csv')
            def_blob_store.blob_service.create_blob_from_path(container_name=blob_container_name,
                                                        blob_name='data/cleaned/'+names[i]+'.csv',
                                                        file_path=names[i]+'.csv')
    else:
        print ('Loading Saved file ...')
        split_data = []
        for (i, name) in enumerate(names):
            temp = pd.read_csv(f'{input_data_ref}/cleaned/{name}.csv', index_col=[0])
            split_data.append(temp)
        split_data_dict = dict(zip(names, [data for data in split_data]))
   
    bert_split_data = []
    names = ['bert_train_split', 'bert_valid_split', 'bert_test_split']
    if not all([is_blob(def_blob_store,'data/processed/'+name+'.csv') for name in names]):
        for (k, target) in copy.deepcopy(split_data_dict).items():
            target['question1'] = target['question1'].apply(lambda x:x[0:max_string_length])
            target['question2'] = target['question2'].apply(lambda x:x[0:max_string_length])
            marked_text = pd.DataFrame('[CLS] ' + target['question1']+ ' [SEP] ' + target['question2']+ ' [SEP] ', columns=['text'])
            marked_text['label'] = target['is_duplicate']
            bert_split_data.append(marked_text)

        for (i, data) in enumerate(bert_split_data):
            data.to_csv(names[i]+'.csv')
            def_blob_store.blob_service.create_blob_from_path(container_name=blob_container_name,
                                                        blob_name='data/processed/'+names[i]+'.csv',
                                                        file_path=names[i]+'.csv')
