import {
  Logs,
  Optimizer,
  OptimizerConstructors,
  Tensor,
  callbacks,
  concat,
  concat3d,
  layers,
  log,
  losses,
  metrics,
  randomNormal,
  sequential,
  stack,
  version,
} from "@tensorflow/tfjs-node";
import { FilesLoader } from "./getData";

export async function runTraining() {
  const { imagens, mascaras } = await FilesLoader.carregarDados({
    diretorioImagens: "./dados/treino/original",
    diretorioMascaras: "./dados/treino/mark",
  });

  if (imagens.length === 0 || mascaras.length === 0) {
    console.error("Nenhum dado encontrado.");
    return;
  }

  const imagensTensor = stack(imagens)
  const labelsTensor = stack(mascaras)
  console.log(imagensTensor.shape);
  console.log(labelsTensor.shape);

  const model = sequential();

  model.add(layers.inputLayer({ inputShape: [1145, 784, 1] }));

  model.add(layers.dense({ units: 16, activation: 'relu' }))
  model.add(layers.dense({ units: 8, activation: 'relu' }))
  model.add(layers.dense({ units: 4, activation: 'sigmoid' }))

  model.compile({
    loss: losses.sigmoidCrossEntropy,
    optimizer: OptimizerConstructors.adam(/*learningRate*/0.001, /*beta1*/0.9, /*beta2*/ 0.999, /*epsilon*/1e-07, ),
    metrics: ['accuracy']
  })
  // Mostra as informações do treinamento
  model.summary();

  const earlyStopping = callbacks.earlyStopping({
    monitor: "loss",
    patience: 10,
  });
  // Traina o modelo
  await model
    .fit(imagensTensor, labelsTensor, {
      epochs: 100,
      batchSize: 15,
      callbacks: [earlyStopping]
    })
    .then((info) => {
      console.log("Precisão final", info.history);
    });

    const saveResult = await model.save("file://models/my-model-10");
    console.log(
      "Modelo salvo:",
      new Date(saveResult.modelArtifactsInfo.dateSaved).toUTCString()
    );

  const prediction = model.predict(randomNormal([1145, 784, 1]));
  (prediction as Tensor).dispose();
}
