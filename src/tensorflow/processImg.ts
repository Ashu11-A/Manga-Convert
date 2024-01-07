import { loaderModel } from "./loader";
import settings from "@/settings.json";
import * as tf from "@tensorflow/tfjs-node";
import sizeOf from "image-size";
import sharp from "sharp";
import { table } from "table";
export async function removeBackground(img: Buffer) {
  const bestModel = settings.tensorflow.bestModel;
  const model = await loaderModel(bestModel);
  if (model === undefined) {
    console.log('Pulando Tensorflow, modelo indefinido...')
    return undefined
  }

  const data = [
    ["Model", "Shape", "dType"],
    [bestModel, model.inputs[0].shape, model.inputs[0].dtype],
  ];
  const dataProcess = [["Size", "Shape", "dType"]];
  const tableScren = table(data);
  console.log(tableScren);

  const { width, height } = sizeOf(img);
  if (width === undefined || height === undefined) return; // Skip invalid images
  // const imgResize = sharp(img).resize(512, 768)

  const inputImage = tf.node.decodeImage(img, 4);
  const preProcessedImage = tf.image.resizeBilinear(inputImage, [768, 512]);
  const inputTensor = preProcessedImage.toFloat(); // Use toFloat() for type conversion

  const normalizedInputs = tf.tidy(() => {
    const dataMax = inputTensor.max();
    const dataMin = inputTensor.min();
    return inputTensor.sub(dataMin).div(dataMax.sub(dataMin));
  });

  const imgTensor = tf.stack([normalizedInputs]);

  const prediction = model.predict(imgTensor);
  dataProcess.push([
    `${height},${width}`,
    String(imgTensor.shape),
    String(imgTensor.dtype),
  ]);
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
      const dataMax = pred3d.max();
      const dataMin = pred3d.min();
      return pred3d.sub(dataMin).div(dataMax.sub(dataMin));
    });
  }

  const pixelsUint8 = await tf.browser.toPixels(pred3d);
  // const pixelsUint8 = new Uint8ClampedArray(pred3d.dataSync().map(value => Math.round(value * 255)))
  // const pixelsUint8 = pred3d.dataSync()

  // Converter Imagem original em buffer
  const image = await sharp(img).toBuffer();
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

  // Aplica a mascara
  const imageProcessed = await sharp(image)
    .composite([{ input: mask, blend: "dest-in", gravity: "northwest" }])
    .png()
    .toBuffer();
  const tableScren2 = table(dataProcess);
  console.log(tableScren2);

  return imageProcessed
}
