import os
import sys
import math
import warnings

from azureml.core import Run
from azureml.core import Workspace, Datastore
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.datastore import Datastore
from azureml.data.data_reference import DataReference

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

import numpy as np
import pandas as pd


cwd = str(os.getcwd())
sys.path.append(cwd)
sys.path.insert(0, cwd)

from BertFineTuning.data_utils import *
from BertFineTuning.model import *
from pytorch_transformers.optimization import AdamW


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
    BFT.train(model_config,ml.bert_train_split,ml.bert_valid_split,epochs=3,print_every=100,validate_at_epoch=0)
    
if (__name__ == "__main__"):
    global run
    run = Run.get_context()
    _params=get_args()
    ws=get_ws(_params)
    
    processed_data_ref=_params.processed_data_ref
    
    def_blob_store = Datastore(ws, 'workspaceblobstore')
    blob_container_name=def_blob_store.container_name
    
#     train()
