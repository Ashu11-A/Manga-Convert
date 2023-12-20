import { load } from "@tensorflow-models/deeplab";
import { Tensor, Tensor3D, Tensor4D, image, node } from "@tensorflow/tfjs-node";
import fs, { readFileSync, readdirSync, statSync } from "fs";
import path from "path";
import { promisify } from "util";

export class FilesLoader {
  public static async carregarDados(options: {
    diretorioImagens: string;
    diretorioMascaras: string;
  }): Promise<{
    imagens: (Tensor3D | Tensor4D)[];
    mascaras: (Tensor3D | Tensor4D)[];
  }> {
    const { diretorioImagens, diretorioMascaras } = options;
    const imagens = [];
    const mascaras = [];

    // Obter lista de nomes de arquivos no diretório de imagens
    const files = readdirSync(diretorioImagens);

    for (const file of files) {
      const filePath = path.join(diretorioImagens, file);
      const stat = statSync(filePath);

      if (stat.isDirectory()) {
        return this.carregarDados({
          diretorioImagens: filePath,
          diretorioMascaras: diretorioMascaras,
        });
      } else if (file.endsWith(".png")) {
        // Carregar imagem original
        const imagemBuffer = readFileSync(filePath);
        const decodedImage = node.decodeImage(imagemBuffer);
        //const imagemResized = image.resizeBilinear(decodedImage, [1150, 784])
        console.log(decodedImage.shape);
        imagens.push(decodedImage);

        // Construir o caminho para a máscara correspondente
        const mascaraBuffer = readFileSync(filePath.replace('original', 'mark'));
        const decodedMascara = node.decodeImage(mascaraBuffer);
        // const newMascara = image.rgbToGrayscale(decodedMascara.slice([0, 0, 0], [-1, -1, 3]))
        //const mascaraResized = image.resizeBilinear(decodedMascara, [1150, 784])
        console.log(decodedMascara.shape);
        mascaras.push(decodedMascara);
      }
    }

    return { imagens, mascaras };
  }
}
