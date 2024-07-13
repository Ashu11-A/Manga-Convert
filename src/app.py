from flask import Flask, request, send_file, Response
from io import BytesIO
import concurrent.futures
from libs.format import format_url
from libs.download import download_image
import asyncio

from models.yolo import process_image

app = Flask(__name__)

async def run_in_executor(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, func, *args)

# Executor para downloads
executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

@app.route('/image')
async def compress_image():
    imageURL = request.args.get('url')
    if not imageURL:
        return 'bandwidth-hero-proxy'

    imageURL = format_url(imageURL)

    if not imageURL:
        return Response("URL inválida!", mimetype='text/plain', headers={"X-Error-Message": f"URL inválida!"})

    # Baixar a imagem em um thread separado
    imageData = await run_in_executor(download_image, imageURL)

    if imageData is None:
        return Response("Erro ao baixar a imagem.", mimetype='text/plain')

    resultPath = await run_in_executor(process_image, imageData)

    if isinstance(resultPath, bytes):
        return send_file(BytesIO(resultPath), mimetype='image/png')

    return send_file(resultPath, mimetype='image/png')

app.run(debug=True, use_reloader=False)
