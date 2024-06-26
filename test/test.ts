import { loadGraphModel } from "@tensorflow/tfjs-converter";
import * as tf from "@tensorflow/tfjs-node"; // Import everything for convenience
import { existsSync, mkdirSync, writeFile } from "fs";
import sizeOf from "image-size";
import path from 'path';
import sharp from "sharp";
import { table } from 'table';
import { FilesLoader } from "./model/getData";

sharp.cache(false)
sharp.concurrency(4)
sharp.simd(true)

export async function testRun() {
  const totalModel = await FilesLoader.countFolders("models");
  const model = await loadGraphModel(
    `file://models/my-model-${totalModel}/model.json`
  );
  const data = [
    ['Model', 'Shape', 'dType'],
    [totalModel, model.inputs[0].shape, model.inputs[0].dtype],
  ];
  const dataProcess = [['Image', 'Size', 'Shape', 'dType']]
  console.log(table(data))
  
  const { imagens } = await FilesLoader.carregarDados({
    diretorioImagens: "./dados/teste/train",
    diretorioMascaras: "./dados/teste/validation",
    onlyTest: true
  });
  
  for (const [currentImage, img] of imagens.entries()) {
    const { width, height } = sizeOf(img);
    if (width === undefined || height === undefined) continue; // Skip invalid images
    // const imgResize = sharp(img).resize(320, 512)
    
    const inputImage  = tf.node.decodeImage(img, 3)
    const preProcessedImage = tf.image.resizeBilinear(inputImage, [1280, 1280])
    const inputTensor = preProcessedImage.toFloat(); // Use toFloat() for type conversion
    
    const normalizedInputs = tf.tidy(() => {
      const dataMax = inputTensor.max();
      const dataMin = inputTensor.min();
      return inputTensor.sub(dataMin).div(dataMax.sub(dataMin));
    });

    console.log(normalizedInputs.min().dataSync()[0], normalizedInputs.max().dataSync()[0])

    
    const imgTensor = tf.stack([normalizedInputs]);
    
    const prediction = model.predict(imgTensor);
    dataProcess.push([String(currentImage), `${height},${width}`, String(imgTensor.shape), String(imgTensor.dtype)])
    let pred3d: tf.Tensor3D;
    
    if (prediction instanceof tf.Tensor) {
      pred3d = prediction.squeeze([0]);
    } else if (typeof prediction === 'object' && prediction !== null && Array.isArray(prediction)) {
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
    
    const pixelsUint8 = await tf.browser.toPixels(pred3d)
    // const pixelsUint8 = new Uint8ClampedArray(pred3d.dataSync().map(value => Math.round(value * 255)))
    // const pixelsUint8 = pred3d.dataSync()

    // Converter Imagem original em buffer
    const image = await sharp(img).toBuffer()
    // Fazer a mascara que será aplicada a imagem original
    const mask = await sharp(pixelsUint8, {
      raw: {
        width: pred3d.shape[1],
        height: pred3d.shape[0],
        channels: 4,
      },
    })
    // .threshold(200)
      .resize({ width, height, fit: 'fill' })
      .png()
      .toBuffer()
    
    // Cria o diretorio para salvar as imagens
    if (!existsSync(path.join('model-test'))) {
      mkdirSync(path.join('model-test'));
    }
    if (!existsSync(path.join('model-test', String(totalModel)))) {
      mkdirSync(path.join('model-test', String(totalModel)));
    }

    // Aplica a mascara
    await sharp(image)
      .composite([{ input: mask, blend: "dest-in", gravity: "northwest" }])
      .png()
      .toFile(`model-test/${totalModel}/prediction-${currentImage}-test.png`)

    writeFile(`model-test/${totalModel}/prediction-${currentImage}-train.png`, img, (err) => {
      if (err) console.error(err)
    });
    writeFile(`model-test/${totalModel}/prediction-${currentImage}-preview.png`, mask, (err) => {
      if (err) console.error(err)
    });
  }
  const tableScren2 = table(dataProcess)
  console.log(tableScren2)
}
testRun();
