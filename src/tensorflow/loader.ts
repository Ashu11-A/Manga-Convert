import { LayersModel, loadLayersModel } from "@tensorflow/tfjs-node";
import { existsSync } from "fs";

export async function loaderModel(model: number): Promise<LayersModel | undefined> {
    try {
        const path = `file://models/my-model-${model}/model.json`
    if (existsSync(path)) {
        return await loadLayersModel(path)
    } else {
        return undefined
    }
    } catch (err) {
        console.log(err)
        return undefined
    }
}