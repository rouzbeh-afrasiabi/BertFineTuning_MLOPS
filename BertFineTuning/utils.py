import os
import sys

cwd = os.getcwd()

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def create_folders(folders):
    for folder in folders:
        if(check_folder(folder)):
            pass
        else:
            os.mkdir(folder)   
            
def check_file(filename,location=cwd):    
    
    return os.path.exists(os.path.join(location,filename)),os.path.join(location,filename)

def check_folder(foldername,location=cwd):    
    
    return os.path.exists(os.path.join(location,foldername))

def find_checkpoints(folder,ext='.pth'):
    checkpoint_files=[]
    for root, dirs, files in os.walk(folder):
        for dir in dirs:
            for child_root, child_dirs, child_files in os.walk(dir):
                for filename in child_files:
                    if(filename.endswith(ext)):
                        if (os.path.join(folder,dir, filename) not in checkpoint_files):
                            checkpoint_files.append(os.path.join(folder,dir, filename))
    return(checkpoint_files)

def get_params(target):
    if(callable(target)):
        try:
            result=list(inspect.signature(target).parameters.keys())
            return(result)
        except:
            return([])
    else:
        return([])
    
data_folder=os.path.join(cwd,'data')
download_folder=os.path.join(cwd,'download')
processed_data_folder=os.path.join(cwd,'data','processed')
original_data_folder=os.path.join(cwd,'data','original')
cleaned_data_folder=os.path.join(cwd,'data','cleaned')
create_folders([data_folder,download_folder,processed_data_folder,cleaned_data_folder])
