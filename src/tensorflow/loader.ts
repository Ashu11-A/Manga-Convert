import { GraphModel, loadGraphModel } from "@tensorflow/tfjs-converter";

export async function loaderModel(
  path: string
): Promise<GraphModel> {
  return await loadGraphModel(`file://${path}/model.json`);
}
