
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

from azureml.core import Run
# from sklearn.utils.class_weight import compute_class_weight
# import spacy
# import warnings
# from download import download
# import numpy as np
# import pandas as pd


def get_args():

  parser = argparse.ArgumentParser()
  parser.add_argument("--input_data_ref",  type=str, dest="input_data_ref")
  args = parser.parse_args()

  return(args)
    

if __name__ == '__main__':
    global run
    run = Run.get_context()
    _params=get_args()
    input_data_ref=_params.input_data_ref
