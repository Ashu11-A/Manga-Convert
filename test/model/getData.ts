import { lstatSync, readFileSync, readdir, readdirSync } from "fs";
import path from "path";

export class FilesLoader {
  public static async carregarDados(options: {
    diretorioImagens: string;
    diretorioMascaras: string;
  }): Promise<{
    imagens: Buffer[];
    mascaras: Buffer[];
  }> {
    const { diretorioImagens, diretorioMascaras } = options;
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
          const mascaraBuffer = readFileSync(
            fullPath.replace("original", "mark")
          );
          mascaras.push(mascaraBuffer);
        }
      });
    }
    scanDirectory(diretorioMascaras);

    return { imagens, mascaras };
  }
  public static countFolders(diretorio: string): number {
    let counter: number = 0;
    readdir(diretorio, (erro, arquivos) => {
      arquivos.forEach((arquivo) => {
        if (lstatSync(arquivo).isDirectory()) {
          counter++;
        }
      });

      console.log(`Total de pastas: ${counter}`);
    });
    return counter
  }
}
