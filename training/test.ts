import {
  Rank,
  Tensor,
  Tensor3D,
  browser,
  loadLayersModel,
  losses,
  metrics,
  node,
  stack,
  train,
} from "@tensorflow/tfjs-node";
import sharp from "sharp";
import { FilesLoader } from "./model/getData";
import { convertToTensor } from "./model/convertToTensor";
import sizeOf from 'image-size'

export async function testRun() {
  const model = await loadLayersModel("file://models/my-model-18/model.json");

  const { imagens, mascaras } = await FilesLoader.carregarDados({
    diretorioImagens: "./dados/teste/original",
    diretorioMascaras: "./dados/teste/mark",
  });

  const { inputs, labels } = convertToTensor({
    inputs: imagens,
    labels: mascaras,
  });

  console.log(inputs.shape, labels.shape);

  model.compile({
    loss: losses.softmaxCrossEntropy,
    optimizer: train.adam(),
    metrics: metrics.categoricalAccuracy,
  });
  
  const evaluation = model.evaluate(inputs, labels);
  if (Array.isArray(evaluation)) {
    evaluation.forEach((metricTensor) => metricTensor.print());
  } else {
    evaluation.print();
  }

  for (const [currentImage, img] of imagens.entries()) {
    const imgResized = [node.decodeImage(img).resizeBilinear([1536, 1024])]
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
