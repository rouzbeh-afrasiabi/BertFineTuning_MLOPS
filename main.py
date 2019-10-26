from azureml.core import Run
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.datastore import Datastore
from azureml.core.dataset import Dataset

import os
import sys
import argparse


cwd = str(os.getcwd())
sys.path.append(cwd)
sys.path.insert(0, cwd)
    

def check_file(filename,location=cwd):    
    
    return os.path.exists(os.path.join(location,filename)),os.path.join(location,filename)

def check_folder(foldername,location=cwd):    
    
    return os.path.exists(os.path.join(location,foldername))


def create_folders(folders):
    for folder in folders:
        if(check_folder(folder)):
            pass
        else:
            os.mkdir(folder)

def get_args():

  parser = argparse.ArgumentParser()

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
        
if __name__ == '__main__':
    global run
    run = Run.get_context()
    _params=get_args()
    ws=get_ws(_params)
    
    datastore_names=list(ws.datastores.keys())
    def_data_store = ws.get_default_datastore()
    def_blob_store = Datastore(ws, datastore_names[1])

    web_path ='https://github.com/rouzbeh-afrasiabi/PublicDatasets/raw/master/train.csv.zip'
    train_zip=Dataset.File.from_files(path=web_path)
    train_zip.register(workspace = ws,
                                 name = 'train_zip_qqp',
                                 description = 'Quora Question Pairs')
