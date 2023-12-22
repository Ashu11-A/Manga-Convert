import {
  Rank,
  Tensor,
  Tensor3D,
  browser,
  node,
  stack
} from "@tensorflow/tfjs-node";
import sizeOf from 'image-size';
import sharp from "sharp";
import { FilesLoader } from "./model/getData";
import converter from '@tensorflow/tfjs-converter'
import { TFSavedModel, loadSavedModel, getMetaGraphsFromSavedModel } from "@tensorflow/tfjs-node/dist/saved_model";

export async function testRun() {
  const model = await converter.GraphModel("models/my-model-19")

  const { imagens } = await FilesLoader.carregarDados({
    diretorioImagens: "./dados/teste/original",
    diretorioMascaras: "./dados/teste/mark",
  });

  for (const [currentImage, img] of imagens.entries()) {
    const imgResized = [node.decodeImage(img).resizeBilinear([768, 512])]
    const imgTensor = stack(imgResized)
    const imgMin = imgTensor.min()
    const imgMax = imgTensor.max()
    const normalizedImg = imgTensor.sub(imgMin).div(imgMax.sub(imgMin))

    const output = model.predict(normalizedImg) as Tensor<Rank>;
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
  }
}
testRun();
