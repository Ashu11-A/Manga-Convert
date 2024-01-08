import { convertSize } from "@/functions/formatBytes";
import { loaderModel } from "./loader";
import settings from "@/settings.json";
import * as tf from "@tensorflow/tfjs-node";
import { existsSync } from "fs";
import sizeOf from "image-size";
import sharp from "sharp";
import { table } from "table";

sharp.cache(false);
sharp.concurrency(4);
sharp.simd(true);
const bestModel = settings.tensorflow.bestModel;
let cachedModel: tf.GraphModel | undefined;

export async function removeBackground(img: Buffer) {
  const memAntes = process.memoryUsage();
  const path = `models/my-model-${bestModel}`;

  if (!existsSync(path)) {
    console.log("Pulando Tensorflow, modelo indefinido...");
    return undefined;
  }

  if (cachedModel === undefined) {
    cachedModel = await loaderModel(bestModel);
  } else {
    console.log("Modelo já carregado!");
  }
  try {
    const { width, height } = sizeOf(img);
    if (width === undefined || height === undefined) return; // Skip invalid images
    if (width < 512 || height < 768) {
      console.log("Imagem muito pequena!");
      return;
    }
    // const imgResize = sharp(img).resize(512, 768)

    const inputImage = tf.node.decodePng(img, 4);
    const resizedImage = tf.image.resizeBilinear(inputImage, [768, 512]);
    const normalizedInputs = tf.tidy(() =>
      resizedImage.sub(resizedImage.min()).div(resizedImage.max().sub(resizedImage.min()))
    );

    const imgTensor = tf.stack([normalizedInputs]);

    const prediction = cachedModel.predict(imgTensor);
    let pred3d: tf.Tensor3D;

    if (prediction instanceof tf.Tensor) {
      pred3d = prediction.squeeze([0]);
    } else if (
      typeof prediction === "object" &&
      prediction !== null &&
      Array.isArray(prediction)
    ) {
      pred3d = prediction[0].squeeze([0]);
    } else {
      throw new Error("prediction deve ser um Tensor ou um Tensor[]");
    }
    if (pred3d.max().dataSync()[0] > 1) {
      pred3d = tf.tidy(() => {
        return pred3d.sub(pred3d.min()).div(pred3d.max().sub(pred3d.min()));
      });
    }

    const pixelsUint8 = await tf.browser.toPixels(pred3d);
    // const pixelsUint8 = new Uint8ClampedArray(pred3d.dataSync().map(value => Math.round(value * 255)))
    // const pixelsUint8 = pred3d.dataSync()

    // Fazer a mascara que será aplicada a imagem original
    const mask = await sharp(pixelsUint8, {
      raw: {
        width: pred3d.shape[1],
        height: pred3d.shape[0],
        channels: 4,
      },
    })
      // .threshold(200)
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

    [mask, invertImage, maskedImage, pixelsUint8].forEach((buffer) =>
      buffer.fill(0)
    );
    [inputImage, normalizedInputs, imgTensor, prediction, pred3d].forEach(
      (tensor) =>
        Array.isArray(tensor)
          ? tensor.map((tensor) => tensor.dispose())
          : tensor.dispose()
    );

    const memDepois = process.memoryUsage();
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

    return imageProcessed;
  } catch (err) {
    console.log('Erro ao processar imagens no Tensorflow');
    return;
  }
}
