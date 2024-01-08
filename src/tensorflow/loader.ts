import { GraphModel, loadGraphModel } from "@tensorflow/tfjs-converter";

export async function loaderModel(
  model: number
): Promise<GraphModel> {
    return await loadGraphModel(`file://models/my-model-${model}/model.json`);
}
