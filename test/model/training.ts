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
import { writeFileSync } from "fs";

export async function runTraining() {
  const { imagens, mascaras } = await FilesLoader.carregarDados({
    diretorioImagens: "./dados/treino/original",
    diretorioMascaras: "./dados/treino/mark",
  });

  if (imagens.length === 0 || mascaras.length === 0) {
    console.error("Nenhum dado encontrado.");
    return;
  }

  // <-- Converte/Embaralha todos os dados para o Tensor -->
  const { inputs, labels } = convertToTensor({
    inputs: imagens,
    labels: mascaras,
  });

  // <-- Cria o modelo -->
  const model = sequential({
    layers: [
      layers.inputLayer({ inputShape: [768, 512, 4]}),
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
      layers.dense({ units: 16, activation: "relu" }),
      layers.dense({ units: 4, activation: "sigmoid" }),
    ],
  });

  model.compile({
    loss: losses.sigmoidCrossEntropy,
    optimizer: train.sgd(0.01),
    metrics: metrics.binaryCrossentropy,
  });

  // <-- Mostra as informações do treinamento -->
  model.summary();

  console.log('Inputs: ', inputs.shape);
  console.log('Labels: ', labels.shape);

  // <-- Isso verifica se está dendo progresso, caso não para o treinamento -->
  const earlyStopping = callbacks.earlyStopping({
    monitor: "binaryCrossentropy",
    patience: 5,
  })

  // <-- Traina o modelo -->
  const result = await model.fit(inputs, labels, {
    epochs: 50,
    batchSize: 1,
    callbacks: [earlyStopping],
  });

  console.log("Precisão final", result.history);

  const totalModal = FilesLoader.countFolders('models')
  model.save(`file://models/my-model-${totalModal}`).then((saveResult) => {
    writeFileSync(`models/my-model-${totalModal}/data.json`, JSON.stringify({
      epochs: result.epoch,
      history: result.history,
      data: result.validationData,
      params: result.params
    }))
    console.log(
      "Modelo salvo:",
      new Date(saveResult.modelArtifactsInfo.dateSaved).toUTCString()
    );
  })

  const prediction = model.predict(randomNormal([1, 768, 512, 4]));
  (prediction as Tensor).print();
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
