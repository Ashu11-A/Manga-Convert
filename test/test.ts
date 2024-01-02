import { loadGraphModel } from "@tensorflow/tfjs-converter";
import * as tf from "@tensorflow/tfjs-node"; // Import everything for convenience
import { writeFile } from "fs";
import sizeOf from "image-size";
import sharp from "sharp";
import { FilesLoader } from "./model/getData";

export async function testRun() {
  const totalModel = await FilesLoader.countFolders("models");
  const model = await loadGraphModel(
    `file://models/my-model-${totalModel}/model.json`
  );

  const { imagens } = await FilesLoader.carregarDados({
    diretorioImagens: "./dados/teste/train",
    diretorioMascaras: "./dados/teste/validation",
  });

  for (const [currentImage, img] of imagens.entries()) {
    const { width, height } = sizeOf(img);
    if (width === undefined || height === undefined) continue; // Skip invalid images
    // const imgResize = sharp(img).resize(512, 768)
    
    const inputImage  = tf.node.decodeImage(img, 4)
    const preProcessedImage = tf.image.resizeBilinear(inputImage, [768, 512])
    const inputTensor = preProcessedImage.toFloat(); // Use toFloat() for type conversion

    const normalizedInputs = tf.tidy(() => {
      const dataMax = inputTensor.max();
      const dataMin = inputTensor.min();
      return inputTensor.sub(dataMin).div(dataMax.sub(dataMin));
    });

    // console.log(normalizedInputs.dataSync())

    const imgTensor = tf.stack([normalizedInputs]);
    console.log(imgTensor.shape, imgTensor.dtype)

    const prediction = model.predict(imgTensor);
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
    // const Uint8Array = new Uint8ClampedArray(pred3d.dataSync().map(value => Math.round(value * 255)))
    // const dataBuffer = pred3d.dataSync()
    const image = await sharp(img).toBuffer()
    const mask = await sharp(pixelsUint8, {
      raw: {
        width: pred3d.shape[1],
        height: pred3d.shape[0],
        channels: 4,
      },
    })
      .threshold(200)
      .resize({ width, height, fit: 'fill' })
      .png()
      .toBuffer()
      // .png()
      // .toFile(`test/prediction-${currentImage}-test.png`);

      await sharp(image)
        .composite([{ input: mask, blend: "dest-in", gravity: "northwest" }])
        // Set the output format and write to a file
        .png()
        .toFile(`test/prediction-${currentImage}-test.png`)

    writeFile(`test/prediction-${currentImage}-train.png`, img, (err) => {
      if (err) console.error(err); // Use console.error for errors
    });
    writeFile(`test/prediction-${currentImage}-preview.png`, mask, (err) => {
      if (err) console.error(err); // Use console.error for errors
    });
  }
}
testRun();
