import os
from pickle import TRUE
import numpy as np
from PIL import Image
from functions.getData import DataLoader
import asyncio
from tqdm import tqdm
import argparse
import io
import cv2

parse = argparse.ArgumentParser(description="Converter Imagens antes do treinamento")
parse.add_argument('--npz', action='store_true', help="Converte em Float32")
parse.add_argument('--png', action='store_true', help="Converte em PNG")
parse.add_argument('--verify', action='store_true', help="Verifica as imagens")
args = parse.parse_args()

def resize_images(path: str):
    image = Image.open(path)

    image = image.resize((256, 512), Image.LANCZOS)
    image = image.convert('RGBA')
    imageInvert = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    
    # Normalize the float32 data to [0, 1]
    # image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))

    # Convert the image to a NumPy array with float32 dtype
    image_array = np.array(image, dtype=np.float32) / 255.0
    imageInvert_array = np.array(imageInvert, dtype=np.float32) / 255.0

    # Save the NumPy array as a compressed npz file
    path = str(path).replace("dados", "dados_cache")

    dirName = os.path.dirname(path)
    fileName = os.path.basename(path)
    print(f"{dirName}/{fileName}")
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        
    # Convert back to 8-bit integer for PNG saving
    # image_array_uint8 = (imageInvert_array * 255).astype(np.uint8)
    # image_png = Image.fromarray(image_array_uint8, mode='RGBA')
    # image_png.save(os.path.join(dirName, fileName.replace(".png", "_float32.png")))

    np.savez_compressed(os.path.join(dirName, fileName.replace(".png", ".npz")), image_array)
    np.savez_compressed(os.path.join(dirName, fileName.replace(".png", "_invert.npz")), imageInvert_array)


path = "dados/treino/train"
loader = DataLoader()

async def processFiles():
    processed = 0
    files = await loader.ListFiles(path)
    if files is not None:
        if args.npz == True:
            for filePath in tqdm(files):
                    if (filePath.endswith('.png')):
                        if (
                            not os.path.exists(filePath.replace("dados", "dados_cache").replace(".png", ".npz")) == True
                            and not os.path.exists(filePath.replace("dados", "dados_cache").replace(".png", "_invert.npz")) == True
                        ):
                            resize_images(filePath)
                            processed += 1
                        else:
                            print(f"Já existe: {filePath}")
                    else:
                        print(f'Arquivo em formato invalido: {filePath}')
        elif args.png == True:
            for filePath in tqdm(files):
                if (not filePath.endswith('.png')):
                    with io.open(filePath, 'rb') as f:
                        image = Image.open(f)
                        image.convert('RGBA')
                        
                        dir = os.path.dirname(filePath)
                        fileName = os.path.basename(filePath).split('.')[0]
                        
                        print(f"{dir}/{fileName}.png")
                        image.save(f"{dir}/{fileName}.png")
                        os.remove(filePath)
                        processed += 1
        elif args.verify == True:
            def join_strings(array):
                return [list(x) for x in zip(*[iter(list(array))]*2)]
            files = join_strings(files)
            for filePath in tqdm(files):
                original = cv2.imread(filePath[0], cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(filePath[1], cv2.IMREAD_GRAYSCALE)
                # Converte a imagem original para escala de cinza, se necessário
                if len(original.shape) == 3:
                    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                if len(mask.shape) == 3:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

                # Redimensiona a máscara para o mesmo tamanho da imagem original
                original.astype(np.uint8)
                mask.astype(np.uint8)

                # Calcula a diferença absoluta entre a imagem original e a máscara
                diff = cv2.absdiff(original, mask)
                diff = diff.astype(np.float32)

                # Calcula a média da diferença
                mean_diff = np.mean(diff)
                if mean_diff > float(50): # type: ignore
                    print(f"A máscara e a imagem original são diferentes. ({mean_diff})")
                    print(filePath)

        else:
            print('Nenhuma ação selecionada')
        print(f'Imagens processadas: {processed}')
    else:
        print('Nenhum Arquivo Encontrado!')

asyncio.run(processFiles())
