import { loadGraphModel } from '@tensorflow/tfjs-converter';
import {
  Rank,
  Tensor,
  Tensor3D,
  browser,
  image,
  node,
  scalar,
  stack,
  tile
} from "@tensorflow/tfjs-node";
import { writeFile } from 'fs';
import sizeOf from 'image-size';
import sharp from "sharp";
import { FilesLoader } from "./model/getData";

export async function testRun() {
  
  const totalModel = await FilesLoader.countFolders('models')
  const model = await loadGraphModel(`file://models/my-model-${totalModel}/model.json`);

  const { imagens } = await FilesLoader.carregarDados({
    diretorioImagens: "./dados/teste/train",
    diretorioMascaras: "./dados/teste/validation",
  });

  for (const [currentImage, img] of imagens.entries()) {
    const inputResized = node.decodeImage(img, 4).resizeBilinear([768, 512])
    const inputTensor = stack([inputResized])
    const normalizedInputs = inputTensor.div(scalar(255.0))

    console.log(normalizedInputs.min().dataSync()[0], normalizedInputs.max().dataSync()[0])
    console.log(normalizedInputs.shape)

    const output = model.predict(normalizedInputs) as Tensor<Rank>;
    const pred3d = output.squeeze([0]) as Tensor3D;
    const pixelsClamped: Uint8ClampedArray = await browser.toPixels(pred3d);
    const pixels = Buffer.from(pixelsClamped.buffer);

    const { width, height } = sizeOf(img)

    console.log(pred3d.shape)
    console.log(width, height)

    if (width === undefined || height === undefined) return

    sharp(pixels, {
      raw: {
        width: pred3d.shape[1],
        height: pred3d.shape[0],
        channels: (pred3d.shape[2] as 3 | 2 | 4 | 1) ?? 1,
      }
    })
      .resize({ width, height })
      .toFile(`test/prediction-${currentImage}.png`);
    writeFile(`test/prediction-${currentImage}-train.png`, img, err => {
      if (err) console.log(err)
    })
  }
}
testRun();
