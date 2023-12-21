import { readFileSync, readdirSync, statSync } from "fs";
import path from "path";

export class FilesLoader {
  public static async carregarDados(options: {
    diretorioImagens: string;
    diretorioMascaras: string;
  }): Promise<{
    imagens: Buffer[]
    mascaras: Buffer[]
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
          imagens.push(imagemBuffer);

          // Construir o caminho para a máscara correspondente
          const mascaraBuffer = readFileSync(
            filePath.replace("original", "mark")
          );
          mascaras.push(mascaraBuffer);
        }
      }

      return { imagens, mascaras };
  }
}
