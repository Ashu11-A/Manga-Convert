import argparse
import tensorflowjs as tfjs
from functions.getData import DataLoader

parser = argparse.ArgumentParser(description='Treinamento de modelo')
parser.add_argument('--model', type=str, help='Se o treinamento deve achar o melhor resultado')
args = parser.parse_args()

if args.model:
    tfjs.converters.convert_tf_saved_model(
        saved_model_dir=f'models/my-model-{args.model}/best_model',
        output_dir=f'models/my-model-{args.model}'
    )
else:
    Loader = DataLoader()
    totalModel = Loader.countFolders('models')
    tfjs.converters.convert_tf_saved_model(
        saved_model_dir=f'models/my-model-{totalModel - 1}/best_model',
        output_dir=f'models/my-model-{totalModel - 1}'
    )