import argparse
import unet.training as unet
import asyncio

parser = argparse.ArgumentParser(description='Treinamento de modelo')
parser.add_argument('--unet', action='store_true', help='Treinar U-Net')
parser.add_argument('--clas', type=str, help='Treinar Classification')
args = parser.parse_args()

async def run():
    if args.unet:
        await unet.runTraining()
    elif args.clas:
        print('Hello world')
        
asyncio.run(run())