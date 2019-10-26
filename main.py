from azureml.core import Run
import argparse

def get_args():

  parser = argparse.ArgumentParser()

  parser.add_argument("--tenant_id",  type=str, dest="tenant_id")
  parser.add_argument("--application_id",  type=str, dest="application_id")
  parser.add_argument("--app_secret",  type=str, dest="app_secret")
  parser.add_argument("--subscription_id",  type=str, dest="subscription_id")
  parser.add_argument("--resource_group",  type=str, dest="resource_group")
  parser.add_argument("--workspace_name",  type=str, dest="workspace_name")
  parser.add_argument("--workspace_region",  type=str, dest="workspace_region")

  args = parser.parse_args()

  tenant_id = args.tenant_id
  application_id = args.application_id
  app_secret = args.app_secret
  subscription_id = args.subscription_id
  resource_group = args.resource_group
  workspace_name = args.workspace_name
  workspace_region = args.workspace_region
  model_name = args.model_name
  cluster_name_cpu = args.cluster_name_cpu
  cluster_name_gpu = args.cluster_name_gpu
  pipeline_experiment_name = args.pipeline_experiment_name
  pipeline_name = args.pipeline_name
      
        
if __name__ == '__main__':
    global run
    run = Run.get_context()
    get_args()
    main()
