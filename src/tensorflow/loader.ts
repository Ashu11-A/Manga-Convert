import { LayersModel, loadLayersModel } from "@tensorflow/tfjs-node";

export async function loaderModel(): Promise<LayersModel> {
    return await loadLayersModel('file://models/my-model-1/model.json')
}