import argparse
import asyncio
from unet.training import unetTraining
from unet.convert import unetConvert

from yolo.test import yoloTest
from yolo.convert import yoloConvert
from yolo.training import yoloTraining
from yolo.better import yoloFindBetter

parser = argparse.ArgumentParser(description='Treinamento de modelo')
parser.add_argument('--unet', action='store_true', help='Treinar Segmentação com U-Net')
parser.add_argument('--yolo', action='store_true', help='Treinar Segmentação com Yolo')

parser.add_argument('--test', action='store_true', help='Testar o Model')
parser.add_argument('--best', action='store_true', help='Isso irá tentar achar os melhores parametros')
parser.add_argument('--convert', action='store_true', help='Coverter um modelo treinado para tensorflow')

parser.add_argument('--size', type=int, help='Definir um tamanho base para o treinamento')
parser.add_argument('--model', type=int, help='Treinar Segmentação com Yolo')

parser.add_argument('--args', action='extend', nargs='+', type=str, help='Definir args de treinamento')
args = parser.parse_args()

async def run():
    if args.unet:
        if args.convert:
            await unetConvert(args.model)
        else:
            await unetTraining(args.args, args.model)
    elif args.yolo:
        if args.test:
            await yoloTest(None)
        elif args.convert:
            await yoloConvert(args.model)
        elif args.best:
            await yoloFindBetter(args.size)
        else:
            await yoloTraining(args.model, args.size, args.args)

asyncio.run(run())