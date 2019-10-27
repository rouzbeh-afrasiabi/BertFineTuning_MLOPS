from azureml.core import Run


def get_args():

  parser = argparse.ArgumentParser()
  parser.add_argument("--def_blob_store",  type=str, dest="def_blob_store")
  args = parser.parse_args()

  return(args)
    

if __name__ == '__main__':
    global run
    run = Run.get_context()
    _params=get_args()
    def_blob_store=_params.def_blob_store
    blob_container_name=def_blob_store.container_name
    
    input_data_ref=def_blob_store.blob_service.get_blob_to_text(container_name=blob_container_name,blob_name='data/original/train.csv')
