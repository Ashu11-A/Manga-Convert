import os
from pathlib import Path
from typing import List, Union

class DataLoader:
    async def LoadFiles(self, markDir: str = '', onlyPath: bool = False) -> tuple[list[bytes], list[bytes]] | tuple[list[str], list[str]] | None:
            images: list[bytes] = []
            masks: list[bytes] = []
            
            imagesPath: list[str] = []
            masksPath: list[str] = []

            def scan_directory(directory: str):
                for entry in os.scandir(directory):
                    if entry.is_dir(follow_symlinks=False):
                        scan_directory(entry.path)
                    elif entry.is_file(follow_symlinks=False) and entry.path.endswith('.npz'):
                        mark_path = Path(entry.path.replace("train", "validation")).with_suffix('.npz')
                        if mark_path.exists():
                            if onlyPath != True:
                                with open(entry.path, 'rb') as file:
                                    image_buffer = file.read()
                                    images.append(image_buffer)
                                with open(mark_path, 'rb') as file:
                                        mask_buffer = file.read()
                                        masks.append(mask_buffer)
                            else: 
                                imagesPath.append(str(entry.path))
                                masksPath.append(str(mark_path))
                        else:
                            print(f"Faltando: {entry.path.replace('train', 'validation')}")
                    else:
                        print(f"File Error: {entry.path}")


            if markDir == '':
                print('Nenhum diretório foi repassado!')
                return None
            else:
                scan_directory(markDir)
                if images or masks:
                    return images, masks
                if imagesPath or masksPath:
                    return imagesPath, masksPath
                    
        
    async def ListFiles(self, markDir: str = '') -> Union[List[str], None]:
        images: List[str] = []

        def scan_directory(directory: str):
            for entry in os.scandir(directory):
                if entry.is_dir(follow_symlinks=False):
                    scan_directory(entry.path)
                elif entry.is_file(follow_symlinks=False) and entry.path.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    # filename, file_extension = os.path.splitext(entry.path)
                    mark_path = Path(entry.path.replace("train", "validation")).with_suffix('.png')
                    if mark_path.exists():
                        images.append(entry.path)
                        images.append(str(mark_path))
                    else:
                        print(f"Faltando: {entry.path.replace('train', 'validation')}")
                else:
                    print(f"File Error: {entry.path}")
        if markDir == '':
            print('Nenhum diretório foi repassado!')
        else:
            scan_directory(markDir)
            return images

    def countFolders(self, directory: str) -> int:
        count = 0
        for entry in os.scandir(directory):
            if entry.is_dir(follow_symlinks=False):
                count += 1
        return count
