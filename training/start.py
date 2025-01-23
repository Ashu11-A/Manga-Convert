import argparse
import asyncio

from yolo.test import segment_images
from yolo.convert import yoloConvert
from yolo.training import yoloTraining
from yolo.better import yoloFindBetter

# Define as opções de argumento para o parser
parser = argparse.ArgumentParser(description='Treinamento de modelo')

# Adiciona os argumentos de linha de comando
parser.add_argument('--unet', action='store_true', help='Treinar Segmentação com U-Net')
parser.add_argument('--yolo', action='store_true', help='Treinar Segmentação com Yolo')

parser.add_argument('--test', action='store_true', help='Testar o Model')
parser.add_argument('--best', action='store_true', help='Isso irá tentar achar os melhores parametros')
parser.add_argument('--convert', action='store_true', help='Converter um modelo treinado para tensorflow')

parser.add_argument('--size', type=int, help='Definir um tamanho base para o treinamento')
parser.add_argument('--model', type=int, help='Treinar Segmentação com Yolo')

parser.add_argument('--args', action='extend', nargs='+', type=str, help='Definir args de treinamento')

# Faz a análise dos argumentos da linha de comando
args = parser.parse_args()


async def run() -> None:
    """Executa as tarefas de treinamento, teste ou conversão do modelo com base nos argumentos fornecidos."""
    if args.yolo:
        if args.test:
            if args.model:
                await segment_images(model_num=args.model)
        elif args.convert:
            if args.model:
                await yoloConvert(args.model)
        elif args.best:
            if args.size:
                await yoloFindBetter(args.size)
        else:
            await yoloTraining(args.model, args.size, args.args)

# Inicia o processo assíncrono
asyncio.run(run())
