from itertools import count
import os
from pathlib import Path
from typing import Dict, List, Union

class DataLoader:
    async def LoadFiles(self, markDir: str = '') -> Union[Dict[str, List[bytes]], None]:
        images: List[bytes] = []
        masks: List[bytes] = []

        def scan_directory(directory: str):
            for entry in os.scandir(directory):
                if entry.is_dir(follow_symlinks=False):
                    scan_directory(entry.path)
                elif entry.is_file(follow_symlinks=False) and entry.path.endswith('.png'):
                    mark_path = Path(entry.path.replace("train", "validation")).with_suffix('.png')
                    if mark_path.exists():
                        with open(entry.path, 'rb') as file:
                            image_buffer = file.read()
                            images.append(image_buffer)
                        with open(mark_path, 'rb') as file:
                                mask_buffer = file.read()
                                masks.append(mask_buffer)


        if markDir == '':
            print('Nenhum diretório foi repassado!')
        else:
            scan_directory(markDir)
            return {'images': images, 'masks': masks}

    def countFolders(self, directory: str) -> int:
        count = 0
        for entry in os.scandir(directory):
            if entry.is_dir(follow_symlinks=False):
                count += 1
        return count
