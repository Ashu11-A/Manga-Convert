import { lstatSync, readFileSync, readdirSync } from "fs";
import path from "path";

export class FilesLoader {
  public static async carregarDados(options: {
    diretorioImagens: string;
    diretorioMascaras: string;
    onlyTest?: boolean
  }): Promise<{
    imagens: Buffer[];
    mascaras: Buffer[];
  }> {
    const { diretorioImagens, diretorioMascaras, onlyTest } = options;
    const imagens: Buffer[] = [];
    const mascaras: Buffer[] = [];

    // Obter lista de nomes de arquivos no diretório de imagens
    function scanDirectory(diretorio: string) {
      readdirSync(diretorio).forEach((file) => {
        const fullPath = path.join(diretorio, file);

        if (lstatSync(fullPath).isDirectory()) {
          scanDirectory(fullPath);
        } else if (path.extname(fullPath) === ".png") {
          // Carregar imagem original
          const imagemBuffer = readFileSync(fullPath);
          imagens.push(imagemBuffer);

          // Construir o caminho para a máscara correspondente
          if (onlyTest !== true) {
            const mascaraBuffer = readFileSync(
              fullPath.replace("train", "validation")
            );
            mascaras.push(mascaraBuffer);
          }
        }
      });
    }
    scanDirectory(diretorioImagens);

    return { imagens, mascaras };
  }
  public static async countFolders(diretorio: string): Promise<number> {
    let counter: number = 0;
    console.log(diretorio)
    const arquivos = readdirSync(diretorio)

    arquivos.forEach((arquivo) => {
      const result = lstatSync(`${diretorio}/${arquivo}`)
      if (result.isDirectory()) {
        counter++
      }
    })
    console.log(`Total de pastas: ${counter}`);
    return counter - 1
  }
}
