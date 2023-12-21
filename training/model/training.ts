import {
  CallbackList,
  Tensor,
  callbacks,
  layers,
  losses,
  metrics,
  randomNormal,
  sequential,
  train,
} from "@tensorflow/tfjs-node";
import { convertToTensor } from "./convertToTensor";
import { FilesLoader } from "./getData";
import { UnresolvedLogs } from "@tensorflow/tfjs-layers/dist/logs";

export async function runTraining() {
  const { imagens, mascaras } = await FilesLoader.carregarDados({
    diretorioImagens: "./dados/treino/original",
    diretorioMascaras: "./dados/treino/mark",
  });

  if (imagens.length === 0 || mascaras.length === 0) {
    console.error("Nenhum dado encontrado.");
    return;
  }

  const { inputs, labels } = convertToTensor({
    inputs: imagens,
    labels: mascaras,
  });

  const model = sequential({
    layers: [
      layers.dense({
        units: 128,
        activation: "relu",
        inputShape: [1568, 784, 4],
      }),
      layers.dropout({ rate: 0.1 }),
      layers.dense({
        units: 64,
        activation: "relu",
      }),
      layers.dropout({ rate: 0.1 }),
      layers.dense({
        units: 32,
        activation: "relu",
      }),
      layers.dropout({ rate: 0.1 }),
      layers.dense({ units: 16, activation: "relu" }),
      layers.dropout({ rate: 0.1 }),
      layers.dense({ units: 4, activation: "sigmoid" }),
    ],
  });

  model.compile({
    loss: losses.sigmoidCrossEntropy,
    optimizer: train.adam(),
    metrics: metrics.categoricalAccuracy,
  });
  // Mostra as informações do treinamento
  model.summary();

  console.log(inputs.shape);
  console.log(labels.shape);
  console.log(`Inputs: ${imagens.length}, Labels: ${mascaras.length}`)

  const earlyStopping = callbacks.earlyStopping({
    monitor: "categoricalAccuracy",
    patience: 5,
  });
  // Traina o modelo
  await model
    .fit(inputs, labels, {
      epochs: 50,
      batchSize: 1,
      callbacks: [earlyStopping],
    })
    .then((info) => {
      console.log("Precisão final", info.history);
    });

  const saveResult = await model.save("file://models/my-model-17");
  console.log(
    "Modelo salvo:",
    new Date(saveResult.modelArtifactsInfo.dateSaved).toUTCString()
  );

  const prediction = model.predict(randomNormal([1, 1568, 784, 4]));
  (prediction as Tensor).dispose();
}

class MyCustomCallback extends CallbackList {
  async onTrainBegin(logs?: UnresolvedLogs) {
    console.log("Início do treinamento");
  }
  async onTrainEnd(logs?: UnresolvedLogs) {
    console.log("Fim do treinamento");
  }
  async onEpochBegin(epoch: number, logs?: UnresolvedLogs) {
    console.log(`Início da época ${epoch}`);
  }
  async onEpochEnd(epoch: number, logs?: UnresolvedLogs) {
    console.log(`Fim da época ${epoch}`);
  }
}
