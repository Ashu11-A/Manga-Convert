import tensorflowjs as tfjs
from model.getData import DataLoader

Loader = DataLoader()

totalModel = Loader.countFolders('models')

tfjs.converters.convert_tf_saved_model(
    saved_model_dir=f'models/my-model-{totalModel - 1}',
    output_dir=f'models/my-model-{totalModel - 1}'
)