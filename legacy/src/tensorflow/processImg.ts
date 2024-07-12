import { convertSize } from "@/functions/formatBytes";
import settings from "@/settings.json";
import "@tensorflow/tfjs-backend-wasm";
import {
  GraphModel,
  Rank,
  Tensor,
  Tensor2D,
  Tensor3D,
  Tensor4D,
  browser,
  cast,
  expandDims,
  image,
  node,
  setBackend,
  slice,
  slice3d,
  stack,
  tensor,
  TensorLike,
  tensor3d,
  tensor4d,
  tidy
} from "@tensorflow/tfjs-node-gpu";
import { createCanvas, ImageData, loadImage } from "canvas";
import { existsSync } from "fs";
import { glob } from 'glob';
import sizeOf from "image-size";
import { join } from "path";
import sharp from "sharp";
import { table } from "table";
import { loaderModel } from "./loader";

const bestModel = settings.tensorflow.bestModel;
const mode = settings.model
let cachedModel: GraphModel;


export async function removeBackground(img: Buffer): Promise<Buffer> {
  const memAntes = process.memoryUsage();
  let path = mode === 'unet' ? 'models/my-model-' : 'runs/segment/train'
  if (existsSync(`${path}${bestModel}`)) {
    path = `${path}${bestModel}`
  } else {
    const dirsLength = (await glob(mode === 'unet' ? 'models/my-model-*' : 'runs/segment/train*/')).length
    path = `${path}${dirsLength !== 1 && mode !== 'unet' ? (dirsLength - 1) : ''}`
  }

  if (mode === 'yolo') path = join(path, 'weights/best_web_model')
  if (cachedModel === undefined) cachedModel = await loaderModel(path)

  try {
    const { width, height } = sizeOf(img);
    if (width === undefined || height === undefined) return img; // Skip invalid images
    // const imgResize = sharp(img).resize(320, 512)

    const normalizeInput = (tensor: Tensor4D | Tensor3D | Tensor<Rank>) => {
      return tensor.sub(tensor.min())
        .div(tensor.max().sub(tensor.min()));
    }

    const convertImage = await sharp(img).removeAlpha().png().toBuffer()

    const decode = node.decodePng(convertImage);
    const resized = image.resizeBilinear(decode, [1280, 1280]);
    const input = cast(resized, 'float32')
    const tensor4D = tensor4d(Array.from(input.dataSync()), [1, 1280, 1280, 3])
    console.log(tensor4D.shape)

    // Fazer a previsão usando o modelo TensorFlow.js
    setBackend("tensorflow");
    const predictionTensor = cachedModel.execute(tensor4D) as Tensor[];
    console.log(predictionTensor)

    // Obter os valores dos pixels da previsão
    const predictionArray = predictionTensor[0].squeeze()
    const newTensor = predictionArray.resizeBilinear([1280, 1280])
    const output = newTensor.slice([0, 0, 0], [1280, 1280, 3])
    //  const normalizedArray = normalizeInput(predictionArray)
    Tensor

    console.log(cachedModel.outputs)
    console.log(newTensor)
    console.log(output)



    // const pixelsUint8 = await browser.toPixels(decodeImage as Tensor3D);
    const encodeImage = await node.encodePng(output as Tensor3D)
    // const imageData = new ImageData(Uint8ClampedArray.from(input.dataSync()), 100, 100);
    // const pixelsUint8 = new Uint8ClampedArray(prediction.dataSync().map(value => Math.round(value * 255)))
    // const pixelsUint8 = pred3d.dataSync()


    // Fazer a mascara que será aplicada a imagem original
    
    const mask = await sharp(encodeImage.buffer)
    // .threshold(125)
      .resize({ width, height, fit: "fill" })
      .png()
      .toBuffer()
    await sharp(mask).toFile('mask.png')
    
    const invertImage = await sharp(img).negate().png().toBuffer();
    await sharp(invertImage).toFile('invertImage.png')

    // Aplica a mascara
    const maskedImage = await sharp(img)
      .composite([
        { input: mask, tile: true, blend: "dest-in", gravity: "center" },
      ])
      .png()
      .toBuffer();
    await sharp(invertImage).toFile('maskedImage.png')

    const imageProcessed = await sharp(invertImage)
      .composite([{ input: maskedImage }])
      .png()
      .toBuffer();
    await sharp(imageProcessed).toFile('imageProcessed.png')
    

    if (settings.debug.any === true) {
      console.log(
        table([
          ["Size", "Shape", "dType"],
          [
            `${height},${width}`,
            String(output.shape),
            String(output.dtype),
          ],
        ])
      );
    }

    [mask, invertImage, maskedImage].forEach((buffer) =>
      buffer.fill(0)
    );
    [output, newTensor, tensor4D, input, resized, decode].forEach((tensor) =>
      Array.isArray(tensor)
        ? tensor.map((tensor) => tensor.dispose())
        : tensor.dispose()
    );

    const memDepois = process.memoryUsage();

    if (settings.debug.memory === true) {
      console.log("Tensorflow Process Images View Memory Usage");
      const tableData = [
        ["Tipo de Memória", "Antes (MB)", "Depois (MB)"],
        ["RSS", convertSize(memAntes.rss), convertSize(memDepois.rss)],
        [
          "Heap Total",
          convertSize(memAntes.heapTotal),
          convertSize(memDepois.heapTotal),
        ],
        [
          "Heap Usado",
          convertSize(memAntes.heapUsed),
          convertSize(memDepois.heapUsed),
        ],
        [
          "External",
          convertSize(memAntes.external),
          convertSize(memDepois.external),
        ],
      ];
      console.log(table(tableData));
    }

    return imageProcessed;
  } catch (err) {
    if (settings.debug.error === true) {
      console.log("Erro ao processar imagens no Tensorflow");
      console.log(err);
    }
    return img
  }
}
