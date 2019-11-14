import os
import sys
import math
import warnings
import re
import uuid

pip_packages=[
              "azureml-sdk==1.0.17", 
              "pandas==0.25.1",
              "numpy==1.17.2",
              "pytorch_transformers==1.0.0",
              "pycm==2.2",
              ]

for item in pip_packages:
    try:
        importlib.import_module(item.split("=")[0])
    except:
        _command="pip install --upgrade "+item
        os.system(_command)


from azureml.core import Run
from azureml.core import Workspace, Datastore
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.datastore import Datastore
from azureml.data.data_reference import DataReference
        
import numpy as np
import pandas as pd


cwd = str(os.getcwd())
sys.path.append(cwd)
sys.path.insert(0, cwd)

from BertFineTuning.data_utils import *
from BertFineTuning.model import *
from pytorch_transformers.optimization import AdamW

from project_config import *

release_id=str(uuid.uuid4())

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data_ref",  type=str, dest="processed_data_ref")
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


def train():
    
    BFT=BertFineTuning()

    BFT.criterion=nn.CrossEntropyLoss()
    BFT.optimizer = AdamW(BFT.parameters_main)
    BFT.scheduler=torch.optim.lr_scheduler.MultiStepLR(BFT.optimizer, milestones=[])

    ml=MultiLoader()
    
    print("Training in progress ...")
    BFT.train(run,ml.bert_train_split,ml.bert_valid_split)
    
if (__name__ == "__main__"):
    global run
    run = Run.get_context()
    _params=get_args()
    ws=get_ws(_params)
    
    data_folder=os.path.join(cwd,'data')
    download_folder=os.path.join(cwd,'download')
    processed_data_folder=os.path.join(data_folder,'processed')
    original_data_folder=os.path.join(data_folder,'original')
    cleaned_data_folder=os.path.join(data_folder,'cleaned')
    create_folders([data_folder,download_folder,processed_data_folder,cleaned_data_folder])

    processed_data_ref=_params.processed_data_ref

    def_blob_store = Datastore(ws, 'workspaceblobstore')
    blob_container_name=def_blob_store.container_name

    blob_container_name=def_blob_store.container_name
    blob_gen=def_blob_store.blob_service.list_blobs(blob_container_name)
    blob_list=[item for item in blob_gen]
    for item in blob_list:
        if (re.match('^'+project_config['project_name']+'\/data\/processed',item.name)):
            _loc=item.name.split(project_config['project_name']+'/')[1]
            def_blob_store.blob_service.get_blob_to_path(container_name=blob_container_name,
                                             blob_name=item.name,
                                            file_path=_loc)   
    print(os.listdir())
    run.add_properties({"release_id":release_id,"project_name":project_config['project_name']})
    train()
