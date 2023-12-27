import sharp from "sharp";

// Função para substituir o fundo branco por preto
async function processImage(inputPath: string, outputPath: string) {
  const image = sharp(inputPath);

  // Converte a imagem para escala de cinza
  const grayscaleImage = await image.greyscale().toBuffer();

  // Aplica um filtro de detecção de quadrinho
  const edgeDetectedImage = await sharp(grayscaleImage).median(3).toBuffer();

  // Cria uma máscara com as bordas em branco
  const mask = await sharp(edgeDetectedImage).negate().toBuffer();

  // Cria uma máscara invertida com as bordas em branco
  const invertedMask = await sharp(edgeDetectedImage)
    .negate()
    .toBuffer();

  // Combina a imagem original com a máscara invertida
  await sharp(inputPath)
    .composite([
        { input: invertedMask, blend: 'dest-over' },
        { input: mask, blend: "over" }
    ])
    .toFile(outputPath);

  console.log("Imagem processada com sucesso!");
}
// Exemplo de uso
const inputImagePath = "./src/3.webp";
const outputImagePath = "./src/imagem_processada.webp";

processImage(inputImagePath, outputImagePath);
