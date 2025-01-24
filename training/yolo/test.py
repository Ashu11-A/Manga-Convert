import cv2 as cv
import numpy as np
import os
from ultralytics import YOLO
from yolo.utils import getModel

def apply_mask(image, mask):
    """Aplica uma máscara binária à imagem, garantindo que os tipos e dimensões estejam corretos."""
    # Garantir que a máscara seja binária (0 ou 255) e do tipo uint8
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    # Verificar se a máscara e a imagem têm o mesmo tamanho
    if image.shape[:2] != mask.shape:
        mask = cv.resize(mask, (image.shape[1], image.shape[0]))
        
    bgr_image = cv.bitwise_and(image, image, mask=mask)
    bgra_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2BGRA)
    bgra_image[:, :, 3] = mask

    # Aplicar a máscara à imagem (bitwise_and espera máscara binária)
    return bgra_image

async def segment_images(model_num=None, image_size=1280):
    """Segmenta imagens usando o modelo YOLOv8 ou YOLOv11 padrão."""
    # Obtém o caminho do modelo
    model_path = getModel(model_num=model_num, find="weights/best.pt")
    images_dir = os.path.abspath("../images")  # Diretório de imagens
    output_dir = os.path.abspath("../output")  # Diretório de saída

    # Garante que a pasta de saída existe
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Carrega o modelo
    print(model_path)
    model = YOLO(model_path)

    # Percorre todas as imagens no diretório
    for image_name in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_name)

        # Ignora arquivos que não sejam imagens
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Carrega a imagem
        image = cv.imread(image_path)
        if image is None:
            print(f"Erro ao carregar a imagem: {image_path}")
            continue

        # Realiza a previsão (segmentação)
        results = model.predict(
          source=image,
          # device="cpu",
          save=False,
          imgsz=[864, 1400],
          conf=0.5,
          task="segment",
          half=True,
          # visualize=True
          augment=True,
          agnostic_nms=True,
          retina_masks=True,
        )

        # Processa os resultados
        for result in results:
            masks = result.masks.data.cpu().numpy()  # Máscaras binárias
            classes = result.boxes.cls.cpu().numpy()  # Classes detectadas
            names = result.names  # Nomes das classes

            # Salva as máscaras processadas
            for i, mask in enumerate(masks):
                class_name = names[int(classes[i])]
                mask = (mask * 255).astype(np.uint8)  # Converte para escala de 0 a 255
                mask_file = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_{class_name}_mask.png")
                cv.imwrite(mask_file, mask)

            # Aplica a máscara à imagem original
            combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
            masked_image = apply_mask(image, combined_mask)
            output_file = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_segmented.png")
            cv.imwrite(output_file, masked_image)

        print(f"Processamento concluído: {image_name}")

if __name__ == "__main__":
    segment_images()
