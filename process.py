from azureml.core import Run



if __name__ == '__main__':
    global run
    run = Run.get_context()
