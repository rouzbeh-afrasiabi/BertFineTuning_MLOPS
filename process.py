from azureml.core import Run


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
