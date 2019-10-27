
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
  parser.add_argument("--ws",  type=str, dest="ws")
  parser.add_argument("--processed_data_ref",  type=str, dest="processed_data_ref")
  args = parser.parse_args()

  return(args)
    

if __name__ == '__main__':
    global run
    run = Run.get_context()
    _params=get_args()
    input_data_ref=_params.input_data_ref
    processed_data_ref=_params.processed_data_ref
