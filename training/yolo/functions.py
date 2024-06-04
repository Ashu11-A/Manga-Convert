from glob import glob
import os

def getModel(model_dir='runs/segment', model_num=None, find: str | None = None):
    if find is None: 
        raise ValueError('Find property was not specified')
    
    modelSubdirs = glob(f'{model_dir}/train*/')
    totalModels = len(modelSubdirs)
    
    # Se não houver modelos treinados
    if totalModels == 0:
        raise ValueError(f'No model subdirectories found in {model_dir}.')
    
    # Caso o modelo seja expecificado
    if model_num:
        model_path = f'{model_dir}/train{model_num}/{find}' if model_num != 0 else f'{model_dir}/train/{find}'
        if os.path.exists(model_path):
            print(f'Convertendo modelo {model_num}')
            return f'{model_dir}/train{model_num}'
        else:
            raise ValueError(f'Model {model_num} not found in {model_dir}.')
    
    # Caso o modelo não seja expecificado
    model_path = f'{model_dir}/train{totalModels}/{find}'
    if os.path.exists(model_path):
        print('Convertendo ultimo modelo')
        return f'{model_dir}/train{totalModels}'
    else:
        model_path = modelSubdirs[totalModels - 1]
        best_model_path = f'{model_path}/{find}'
        if os.path.exists(best_model_path):
            print(f'Convertendo modelo {model_path}')
            return model_path
    
    raise ValueError(f'No best.pt model files found in {model_dir}/{model_path} or subdirectories')