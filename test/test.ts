import { loadGraphModel } from '@tensorflow/tfjs-converter';
import {
  Rank,
  Tensor,
  Tensor3D,
  browser,
  node,
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
    diretorioImagens: "./dados/teste/original",
    diretorioMascaras: "./dados/teste/mark",
  });

  for (const [currentImage, img] of imagens.entries()) {
    const inputResized = [node.decodeImage(img).resizeBilinear([768, 512])]
    let inputTensor = tile(stack(inputResized), [1, 1, 1, 4])
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const normalizedInputs = inputTensor
    .sub(inputMin)
    .div(inputMax.sub(inputMin));

    console.log(normalizedInputs.shape)

    const output = model.predict(normalizedInputs) as Tensor<Rank>;
    const pred3d = output.squeeze([0]) as Tensor3D;
    const pixelsClamped: Uint8ClampedArray = await browser.toPixels(pred3d);
    const pixels = Buffer.from(pixelsClamped.buffer);

    const { width, height } = sizeOf(img)

    if (width === undefined || height === undefined) return

    sharp(pixels, {
      raw: {
        width: pred3d.shape[1],
        height: pred3d.shape[0],
        channels: (pred3d.shape[2] as 3 | 2 | 4 | 1) ?? 1,
      }
    })
      .resize({ width, height })
      .png()
      .toFile(`test/prediction-${currentImage}.png`);
    writeFile(`test/prediction-${currentImage}-original.png`, img, err => {
      if (err) console.log(err)
    })
  }
}
testRun();
