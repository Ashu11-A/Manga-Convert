import os
from pickle import TRUE
import numpy as np
from PIL import Image
from model.getData import DataLoader
import asyncio

def resize_images(path: str):
    image = Image.open(path)

    image = image.resize((512, 768), Image.LANCZOS)
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


path = "dados/treino"
loader = DataLoader()

async def processFiles():
    files = await loader.ListFiles(path)
    if files is not None:
        for filePath in files:
            if (
                not os.path.exists(filePath.replace("dados", "dados_cache").replace(".png", ".npz")) == True
                and not os.path.exists(filePath.replace("dados", "dados_cache").replace(".png", "_invert.npz")) == True
            ):
                resize_images(filePath)
            else:
                print(f"Já existe: {filePath}")
    else:
        print('Nenhum Arquivo Encontrado!')

asyncio.run(processFiles())
