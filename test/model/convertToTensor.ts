import {
  Rank,
  Tensor,
  node,
  stack,
  tensor1d,
  tidy,
  util
} from "@tensorflow/tfjs-node";

export function convertToTensor(options: {
  inputs: Buffer[]
  labels: Buffer[]
}): {
  inputs: Tensor<Rank>;
  labels: Tensor<Rank>;
  inputMax: Tensor<Rank>;
  inputMin: Tensor<Rank>;
  labelMax: Tensor<Rank>;
  labelMin: Tensor<Rank>;
} {
  const { inputs, labels } = options;

  return tidy(() => {

    // <-- Faz o redimencionamento da imagens -->
    const inputResized = inputs.map((img) => {
        return node.decodeImage(img).resizeBilinear([512, 256])
    })
    const labelResized = labels.map((img) => {
        return node.decodeImage(img).resizeBilinear([512, 256])
    })

    let inputTensor = stack(inputResized);
    let labelsTensor = stack(labelResized);

    // <-- Embaralhamento -->
    const indices = util.createShuffledIndices(inputTensor.shape[0])
    const tensorIndices = tensor1d(Array.from(indices), 'int32'); 

    inputTensor = inputTensor.gather(tensorIndices)
    labelsTensor = labelsTensor.gather(tensorIndices)

    // <-- Converte para o espectro [0, 1] -->
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelsTensor.max();
    const labelMin = labelsTensor.min();

    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin));
    const normalizedLabels = labelsTensor
      .sub(labelMin)
      .div(inputMax.sub(inputMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    };
  });
}
