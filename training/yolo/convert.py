from ultralytics import YOLO
from yolo.utils import getModel
import yaml
import os

async def yoloConvert(model_num: int | None=None):
  modelPath = getModel(model_num=model_num, find='weights/best.pt')
  modelSize = yaml.safe_load(open(os.path.join(modelPath, 'args.yaml')))['imgsz']

  if not isinstance(modelSize, list) and not isinstance(modelSize, int):
    raise ValueError(f'Não foi possivel determinar o tamanho do modelo: {modelPath}')

  model = YOLO(os.path.join(modelPath, 'weights/best.pt'), task='segment')

  # Export the model to TF SavedModel format
  if not os.path.exists(os.path.join(modelPath, 'weights/best_web_model')):
    # format: Formato de destino para o modelo exportado, como por exemplo 'onnx', 'torchscript', 'tensorflow'ou outros, definindo a compatibilidade com vários ambientes de implantação.
    # imgsz: Tamanho de imagem pretendido para a entrada do modelo. Pode ser um número inteiro para imagens quadradas ou uma tupla (height, width) para dimensões específicas.
    # keras: Permite exportar para o formato Keras para TensorFlow SavedModel , proporcionando compatibilidade com TensorFlow servindo e APIs.
    # int8: Ativa a quantização INT8, comprimindo ainda mais o modelo e acelerando a inferência com uma perda mínima de precisão, principalmente para dispositivos de ponta.
    model.export(format='tfjs', imgsz=modelSize, keras=True)  # creates '/yolov8n_web_model'

  
  return modelPath