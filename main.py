from azureml.core import Run
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.datastore import Datastore
from azureml.core.dataset import Dataset
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
from azureml.core import Experiment
from azureml.pipeline.core.graph import PipelineParameter

 
import os
import sys
import requests
import shutil
import zipfile
import argparse
import json
try:
    from download import download
except:
    os.system("pip install download")


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
    auth_params=get_args()
    ws=get_ws(auth_params)
    
    datastore_names=list(ws.datastores.keys())
    def_data_store = ws.get_default_datastore()
    def_blob_store = Datastore(ws, "workspaceblobstore")
    
    data_temp_folder=os.path.join(cwd,"data_temp")
    create_folders([data_temp_folder])
    
    dataset={'dataset':"https://github.com/rouzbeh-afrasiabi/PublicDatasets/raw/master/train.csv.zip"}
    
    
    toDownload=[dataset]
    download_files(toDownload,data_temp_folder)
    
    zip_file = zipfile.ZipFile(os.path.join(data_temp_folder,"train.csv.zip"), 'r')
    zip_file.extractall(data_temp_folder)
    zip_file.close() 
     
    def_blob_store.upload_files(
                                [os.path.join(data_temp_folder,"train.csv")],
                                target_path="data/original/",
                                overwrite=True)
    
    cluster_name = "cpucluster"
    
    try:
        compute_target_cpu = ComputeTarget(workspace=ws, name=cluster_name)
    except ComputeTargetException:
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2', 
                                                               max_nodes=1,
                                                               min_nodes=1)
        compute_target_cpu = ComputeTarget.create(ws, cluster_name, compute_config)
        compute_target_cpu.wait_for_completion(show_output=True)
    

    input_data_ref = DataReference(
                            datastore=def_blob_store,   
                            data_reference_name="input_data_ref",
                            path_on_datastore="data/")
    
    processed_data_ref = PipelineData("processed_data_ref", datastore=def_blob_store)
    
    pipeline_params=[]    
    for k,v in vars(auth_params).items():
     pipeline_params.append("--"+k)
     pipeline_params.append(PipelineParameter(name=k,default_value=v))
     
    pipeline_params+=["--processed_data_ref",processed_data_ref]
    process_step = PythonScriptStep(script_name="process.py",
                                   arguments=pipeline_params,
                                   inputs=[input_data_ref],
                                    outputs=[processed_data_ref],
                                   compute_target=compute_target_cpu,
                                   source_directory='./')

    run_config = RunConfiguration()
    run_config.environment.docker.enabled = True
    run_config.environment.docker.base_image = DEFAULT_CPU_IMAGE
    run_config.environment.python.user_managed_dependencies = False
    pip_packages=[
                "azureml-sdk==1.0.17", "scikit-learn==0.21.3",
                "download==0.3.4", "pandas==0.25.1",
                "spacy==2.1.4", "numpy==1.17.2"]
    run_config.environment.python.conda_dependencies = CondaDependencies.create(pip_packages=pip_packages)
    
    pipeline = Pipeline(workspace=ws, steps=[process_step])
    pipeline_run_first = Experiment(ws, 'test_exp_1').submit(pipeline)
    pipeline_run_first.wait_for_completion()
