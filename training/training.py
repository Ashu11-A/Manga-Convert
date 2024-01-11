import argparse
import unet.training as unet
import asyncio
import subprocess

parser = argparse.ArgumentParser(description='Treinamento de modelo')
parser.add_argument('--unet', action='store_true', help='Treinar U-Net')
parser.add_argument('--clas', type=str, help='Treinar Classification')
args = parser.parse_args()

async def run():
    if args.unet:
        try:
            subprocess.run(['python', 'training/unet/training.py'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Erro ao executar o arquivo externo: {e}")
    elif args.clas:
        print('Hello world')
        
asyncio.run(run())