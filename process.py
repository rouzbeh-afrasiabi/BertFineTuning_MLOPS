
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

pip_packages=[
              "azureml-sdk==1.0.17", "scikit-learn==0.21.3",
              "download==0.3.4", "pandas==0.25.1",
              "spacy==2.1.4", "numpy==1.17.2"
              ]

for item in pip_packages:
  try:
    importlib.import_module(item.split("=")[0])
  except:
    _command="pip install "+item
    os.system(_command)

    
from sklearn.utils.class_weight import compute_class_weight
import spacy
from download import download
import numpy as np
import pandas as pd


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
    

if __name__ == '__main__':
    global run
    run = Run.get_context()
    _params=get_args()
    processed_data_ref=_params.processed_data_ref
