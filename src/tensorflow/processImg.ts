import { convertSize } from "@/functions/formatBytes";
import settings from "@/settings.json";
import "@tensorflow/tfjs-backend-wasm";
import {
  GraphModel,
  Rank,
  Tensor,
  Tensor3D,
  browser,
  image,
  node,
  setBackend,
  stack,
  tidy
} from "@tensorflow/tfjs-node-gpu";
import { existsSync, readdirSync, statSync } from "fs";
import sizeOf from "image-size";
import sharp from "sharp";
import { table } from "table";
import { loaderModel } from "./loader";

const bestModel = settings.tensorflow.bestModel;
const mode = settings.model
let cachedModel: GraphModel;

function getNumberOfFolders(path: string) {
  const files = readdirSync(path);
  const folders = files.filter((file) => statSync(path + "/" + file).isDirectory());
  return folders.length;
}


export async function removeBackground(img: Buffer): Promise<Buffer> {
  const memAntes = process.memoryUsage();
  let path = mode === 'unet' ? 'models/my-model-' : 'runs/segment/train'
  if (existsSync(`${path}${bestModel}`)) {
    path = `${path}${bestModel}`
  } else {
    path = `${path}${getNumberOfFolders('models') - 1}`
  }

  if (cachedModel === undefined) cachedModel = await loaderModel(path)

  try {
    const { width, height } = sizeOf(img);
    if (width === undefined || height === undefined) return img; // Skip invalid images
    // const imgResize = sharp(img).resize(320, 512)

    const imgTensor = tidy(() => {
      const inputImage = node.decodeImage(img, 3);
      const resizedImage = image.resizeBilinear(inputImage, [512 ,320]);
      const normalizedInputs = resizedImage
        .sub(resizedImage.min())
        .div(resizedImage.max().sub(resizedImage.min()));

      return stack([normalizedInputs]);
    });

    const predict = async (input: Tensor<Rank>) => {
      setBackend("tensorflow");
      const output = cachedModel.execute(input) as Tensor3D;
      return output;
    };

    let prediction = await predict(imgTensor);
    prediction = prediction.squeeze([0]);
    if (prediction.max().dataSync()[0] > 1) {
      prediction = prediction
        .sub(prediction.min())
        .div(prediction.max().sub(prediction.min()));
    }

    const pixelsUint8 = await browser.toPixels(prediction as Tensor3D);
    // const pixelsUint8 = new Uint8ClampedArray(pred3d.dataSync().map(value => Math.round(value * 255)))
    // const pixelsUint8 = pred3d.dataSync()

    // Fazer a mascara que será aplicada a imagem original
    const mask = await sharp(pixelsUint8, {
      raw: {
        width: prediction.shape[1] as number,
        height: prediction.shape[0],
        channels: 4,
      },
    })
      // .threshold(125)
      .resize({ width, height, fit: "fill" })
      .png()
      .toBuffer();

    const invertImage = await sharp(img).negate().png().toBuffer();

    // Aplica a mascara
    const maskedImage = await sharp(img)
      .composite([
        { input: mask, tile: true, blend: "dest-in", gravity: "center" },
      ])
      .png()
      .toBuffer();

    const imageProcessed = await sharp(invertImage)
      .composite([{ input: maskedImage }])
      .png()
      .toBuffer();

    if (settings.debug.any === true) {
      console.log(
        table([
          ["Size", "Shape", "dType"],
          [
            `${height},${width}`,
            String(imgTensor.shape),
            String(imgTensor.dtype),
          ],
        ])
      );
    }

    [mask, invertImage, maskedImage, pixelsUint8].forEach((buffer) =>
      buffer.fill(0)
    );
    [imgTensor, prediction].forEach((tensor) =>
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
