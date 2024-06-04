import tensorflowjs as tfjs
import os
from glob import glob

async def unetConvert(model: int | None=None):
    if model is not None:
        modelPath = f'models/my-model-{model}'
        if os.path.exists(modelPath) is False:
            raise(f'Not found model {modelPath}')

        tfjs.converters.convert_tf_saved_model(
            saved_model_dir=f'models/my-model-{model}/best_model',
            output_dir=modelPath
        )
    else:
        models = glob('models/my-model-*')
        totalModels = len(models)
        modelPath = f'models/my-model-{totalModels}'
        if os.path.exists(modelPath) is False:
            raise(f'Not found model {modelPath}')

        tfjs.converters.convert_tf_saved_model(
            saved_model_dir=f'models/my-model-{totalModels - 1}/best_model',
            output_dir=f'models/my-model-{totalModels - 1}'
        )