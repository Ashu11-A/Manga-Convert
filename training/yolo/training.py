from ultralytics import YOLO, checks
from yolo.utils import getModel
import os

checks()

async def yoloTraining (model: None | int = None, size: int | list[int] = 1280, args: list[str] = []):
  # Load the YOLOv8 model
  if model is not None:
    modelPath = getModel(model_num=model, find='weights/best.pt')
    print(f'⚠️ Treinando modelo com base no modelo {model}, que setá em {modelPath}')
    model = YOLO(f'{modelPath}/weights/best.pt', task='segment')
  else:
    model = YOLO("yolov8s-seg", task='segment')

  # https://docs.ultralytics.com/pt/usage/cfg/#train-settings
  # https://medium.com/@nainaakash012/when-does-label-smoothing-help-89654ec75326
  # https://github.com/orgs/ultralytics/discussions/10320
  model.train(
    data=os.path.abspath("../dataset/yolo/data.yaml"),
    cfg=os.path.abspath("yolo/best_hyperparameters.yaml"),
    patience=25,
    epochs=1000,
    batch=2,
    imgsz=size,
    # rect=True, # Modo Retangular
    # resume=True
    cache=True,
    optimizer="AdamW",
    # seed=369
  )

  # Export the model to TF SavedModel format
  # format: Formato de destino para o modelo exportado, como por exemplo 'onnx', 'torchscript', 'tensorflow'ou outros, definindo a compatibilidade com vários ambientes de implantação.
  # imgsz: Tamanho de imagem pretendido para a entrada do modelo. Pode ser um número inteiro para imagens quadradas ou uma tupla (height, width) para dimensões específicas.
  # keras: Permite exportar para o formato Keras para TensorFlow SavedModel , proporcionando compatibilidade com TensorFlow servindo e APIs.
  # int8: Ativa a quantização INT8, comprimindo ainda mais o modelo e acelerando a inferência com uma perda mínima de precisão, principalmente para dispositivos de ponta.
  model.export(format="onnx", imgsz=size, half=True)  # creates '/yolov8n_web_model'