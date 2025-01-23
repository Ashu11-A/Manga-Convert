from glob import glob
import os

def getModel(model_dir=None, model_num=None, find=None):
    if model_dir is None:
        model_dir = os.path.abspath('../runs/segment')
    if find is None:
        raise ValueError('Find property was not specified')
    
    print(f'Searching in directory: {model_dir}')
    modelSubdirs = sorted(glob(f'{model_dir}/train*/'))  # Sort for consistency
    totalModels = len(modelSubdirs)
    print(f'Found {totalModels} model subdirectories: {modelSubdirs}')
    
    if totalModels == 0:
        raise ValueError(f'No model subdirectories found in {model_dir}.')
    
    if model_num is not None:
        # Handle case where model_num is 0
        if model_num == 0:
            model_path = os.path.join(model_dir, 'train', find)
            if os.path.exists(model_path):
                print(f'Using model 0 (directory "train"): {model_path}')
                return model_path
            else:
                raise ValueError(f'Model 0 not found in {model_dir}.')
        else:
            model_path = os.path.join(model_dir, f'train{model_num}', find)
            if os.path.exists(model_path):
                print(f'Using specified model {model_num}: {model_path}')
                return model_path
            else:
                raise ValueError(f'Model {model_num} not found in {model_dir}.')
    
    # Default to the last model if model_num is not specified
    model_path = os.path.join(modelSubdirs[-1], find)
    if os.path.exists(model_path):
        print(f'Using latest model in directory: {modelSubdirs[-1]}')
        return modelSubdirs[-1]
    
    raise ValueError(f'No {find} file found in {model_dir} or its subdirectories.')
