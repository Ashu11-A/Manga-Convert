import { Tensor2D, Tensor3D, Tensor4D, argMax, browser, concat, expandDims, loadLayersModel, metrics, slice, squeeze, stack } from "@tensorflow/tfjs-node";
import { FilesLoader } from "./model/getData";
import sharp, { Sharp } from 'sharp'
import { PNG } from 'pngjs'
import { createWriteStream } from "fs";

export async function testRun() {
    const model = await loadLayersModel('file://models/my-model-11/model.json')

    const { imagens, mascaras} = await FilesLoader.carregarDados({
        diretorioImagens: './dados/teste/original',
        diretorioMascaras: './dados/teste/mark'
    })


    const imagensConvert = imagens.map(img => {
        return img.resizeBilinear([1145, 784])
    })

    const mascarasConvert = mascaras.map(img => {
        return img.resizeBilinear([1145, 784])
    })

    const imagensTensor = stack(imagensConvert)
    const labelsTensor = stack(mascarasConvert)

    console.log(imagensTensor.shape)
    console.log(labelsTensor.shape)

    model.compile({ loss: 'binaryCrossentropy', optimizer: 'adam', metrics: [metrics.categoricalAccuracy] })

    const evaluation = model.evaluate(imagensTensor, labelsTensor)
    if (Array.isArray(evaluation)) {
        evaluation.forEach(metricTensor => metricTensor.print())
    } else {
        evaluation.print()
    }

    const output = model.predict(imagensTensor)
    let image: Sharp

    console.log(output)
    if (Array.isArray(output)) {
        output.map(async (value) => {
            const data = await value.data()
            const buffer = new Uint8Array(data)
            image = sharp(buffer, {
                raw: {
                    width: 784,
                    height: 1145,
                    channels: 2
                }
            })
            image.toFile('prediction.png')
        })
    } else {
        if (output.rank === 4) {
            let pred3d = squeeze(output, [0]) as Tensor3D; // converte 4d em 3d
            let pixelsClamped: Uint8ClampedArray = await browser.toPixels(pred3d);
            let pixels = Buffer.from(pixelsClamped.buffer);
            let png = new PNG({
                width: pred3d.shape[1],
                height: pred3d.shape[0],
                filterType: -1
            });
            png.data = pixels;
            png.pack().pipe(createWriteStream('prediction.png'));
        } else {
            console.error('The prediction is not a 4D tensor');
        }
    }
}
testRun()