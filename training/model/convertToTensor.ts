import {
  Rank,
  Tensor,
  Tensor3D,
  Tensor4D,
  node,
  stack,
  tidy,
  util,
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
    util.shuffle(inputs);
    util.shuffle(labels);

    const inputResized = inputs.map((img) => {
        return node.decodeImage(img).resizeBilinear([1568, 784])
    })
    const labelResized = labels.map((img) => {
        return node.decodeImage(img).resizeBilinear([1568, 784])
    })

    const inputTensor = stack(inputResized);
    const labelsTensor = stack(labelResized);

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
