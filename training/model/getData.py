from itertools import count
from os import scandir
from pathlib import Path
from typing import Dict, List, Union

class DataLoader:
    async def LoadFiles(self, markDir: str = '') -> Union[Dict[str, List[bytes]], None]:
        
        imagens: list[bytes] = []
        mascaras: list[bytes] = []
        
        def scanDirectory(dir: str):
            for entry in scandir(dir):
                if entry.is_dir(follow_symlinks=False): # Verifica se o arquivo é um diretorio, se sim reescaneia os itens dessa pasta
                    scanDirectory(entry.path)
                elif entry.is_file(follow_symlinks=False) and entry.path.endswith('.png'):
                    with open(entry.path, 'rb') as file:
                        imageBuffer = file.read()
                        imagens.append(imageBuffer)
                        file.close()
                        
                        markPath = Path(entry.path.replace("original", "mark")).with_suffix('.png')
                        if markPath.exists():  
                            with open(markPath, 'rb') as file:
                                markBuffer = file.read()
                                mascaras.append(markBuffer)
                                file.close
        
        if markDir == '':
            print('Nenhum diretório foi repassado!')
        else:                        
            scanDirectory(markDir)
            return { 'imagens': imagens, 'mascaras': mascaras }
    def countFolders(self, dir: str) -> int:
        count: int = 0
        for entry in scandir(dir):
            if entry.is_dir(follow_symlinks=False):
                count += 1
        return count