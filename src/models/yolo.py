from io import BytesIO
from os import path
from PIL import Image
import numpy as np
from ultralytics import YOLO

model = YOLO(model=path.join('../models/yolo/v1/weights/best.pt'), task='segment')

def process_image(image_data):
    try:
        origemImage = BytesIO(image_data)
        image = Image.open(origemImage).convert('RGB')

        # # Verifique o tamanho da imagem
        # if image.width < 500 or image.height < 500:
        #     print('Imagem muito pequena')
        #     return image_data  # Retorna a imagem original se for pequena

        imageArray = np.array(image)
        results = model.predict(imageArray, show_boxes=True, retina_masks=True, agnostic_nms=True)

        if not results:
            print('Sem resultados de YOlo')
            return image_data  # Retorna a imagem original se não houver resultados

        for result in results:
            annotatedImg = result.plot()
            resultImage = Image.fromarray(annotatedImg)

            # Salva a imagem anotada em um buffer
            imgIO = BytesIO()
            resultImage.save(imgIO, format='PNG')
            imgIO.seek(0)
            
            return imgIO

    except Exception as e:
        print(f"Erro ao processar a imagem: {e}")
        return image_data  # Retorna a imagem original em caso de erro
    