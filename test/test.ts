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

    const inputResized = tf.node.decodeImage(img, 4).resizeBilinear([768, 512]);
    const inputTensor = inputResized.toFloat(); // Use toFloat() for type conversion

    console.log(inputTensor.shape)

    const normalizedInputs = tf.tidy(() => {
      const dataMax = inputTensor.max();
      const dataMin = inputTensor.min();
      return inputTensor.sub(dataMin).div(dataMax.sub(dataMin));
    });

    const imgTensor = tf.stack([normalizedInputs]);

    const prediction = model.predict(imgTensor);
    let pred3d: tf.Tensor3D;
    
    if (prediction instanceof tf.Tensor) {
      pred3d = prediction.squeeze([0]);
    } else if (typeof prediction === 'object' && prediction !== null && Array.isArray(prediction)) {
      pred3d = prediction[0].squeeze([0]);
    } else {
      throw new Error("prediction deve ser um Tensor ou um Tensor[]");
    }
    const pixelsUint8 = await tf.browser.toPixels(pred3d); // Use toPixels() directly for Uint8Array

    await sharp(pixelsUint8, {
      raw: {
        width: pred3d.shape[1],
        height: pred3d.shape[0],
        channels: (pred3d.shape[2] as 3 | 2 | 4 | 1) || 1,
      },
    })
      .resize({ width, height })
      .png()
      .toFile(`test/prediction-${currentImage}-test.png`);

    // No need for separate sharp calls with Buffer and Uint8Array

    writeFile(`test/prediction-${currentImage}-train.png`, img, (err) => {
      if (err) console.error(err); // Use console.error for errors
    });
  }
}
testRun();
