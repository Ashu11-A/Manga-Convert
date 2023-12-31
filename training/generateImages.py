import os
import numpy as np
from PIL import Image
from model.getData import DataLoader
import asyncio

def resize_images(path: str):
    image = Image.open(path)
    image = image.resize((512, 768), Image.LANCZOS)
    image = image.convert('RGBA')

    # Convert the image to a NumPy array with float32 dtype
    image_array = np.array(image, dtype=np.float32) / 255.0

    # Save the NumPy array as a compressed npz file
    path_str = path  # Converte Path para string
    path = path_str.replace("dados", "dados_cache")

    dirName = os.path.dirname(path)
    fileName = os.path.basename(path)
    print(f"{dirName}/{fileName}")
    if not os.path.exists(dirName):
        os.makedirs(dirName)

    np.savez_compressed(os.path.join(dirName, fileName.replace(".png", ".npz")), image_array)


path = "dados/treino"
loader = DataLoader()

async def processFiles():
    files = await loader.ListFiles(path)
    if files is not None:
        [resize_images(filePath) for filePath in files]
    else:
        print('Nenhum Arquivo Encontrado!')

asyncio.run(processFiles())


# import os
# import numpy as np
# from PIL import Image
# from model.getData import DataLoader
# import asyncio

# def resize_images(path: str):
#     image = Image.open(path)
#     image = image.resize((512, 768), Image.LANCZOS)
#     image = image.convert('RGBA')

#     # Convert the image to a NumPy array with float32 dtype
#     image_array = np.array(image, dtype=np.float32)

#     # Normalize the float32 data to [0, 1]
#     image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))

#     # Save the normalized float32 data as a PNG image
#     path_str = path  # Converte Path para string
#     path = path_str.replace("dados", "dados_cache")

#     dirName = os.path.dirname(path)
#     fileName = os.path.basename(path)
#     print(f"{dirName}/{fileName}")
#     if not os.path.exists(dirName):
#         os.makedirs(dirName)

#     # Convert back to 8-bit integer for PNG saving
#     image_array_uint8 = (image_array * 255).astype(np.uint8)
#     image_png = Image.fromarray(image_array_uint8, mode='L')
#     image_png.save(os.path.join(dirName, fileName.replace(".png", "_float32.png")))


# path = "dados/treino"
# loader = DataLoader()

# async def processFiles():
#     files = await loader.ListFiles(path)
#     if files is not None:
#         [resize_images(filePath) for filePath in files]
#     else:
#         print('Nenhum Arquivo Encontrado!')

# asyncio.run(processFiles())
