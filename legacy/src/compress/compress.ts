import sharp from "sharp";
import { Request, Response } from "express";
import { redirect } from "../proxy/redirects";
import { shouldCompress } from "./shouldCompress";
import { bypass } from "@/proxy/bypass";
import { table } from "table";
import { convertSize } from "@/functions/formatBytes";
import settings from "@/settings.json";
import { send } from "@/proxy/send";

export async function compress(
  req: Request,
  res: Response,
  input: Buffer
): Promise<Buffer | undefined> {
  const memAntes = process.memoryUsage();
  console.log("Iniciando compressão...");

  try {
    const output = await sharp(input)
      .grayscale(req.params.grayscale ? true : false)
      .png({
        quality: parseInt(req.params.quality) || 80,
      })
      .toBuffer();

    const originalSize = parseInt(req.params.originSize);
    const compressedSize = output.length;

    if (settings.debug.any === true) {
      const infoTable = [
        ["Input", "Tensor", "Output"],
        [
          convertSize(originalSize),
          convertSize(parseInt(req.params.tensorflowSize)),
          convertSize(compressedSize),
        ],
      ];
      console.log(table(infoTable));
    }
    if (compressedSize >= originalSize) {
      console.log("Imagem maior que a comprimida.");
      return input;
    } else {
      console.log("Imagem comprimida com sucesso.");
      return output;
    }
  } catch (err) {
    if (settings.debug.error === true) {
      console.log("Erro na compactação");
      console.log(err);
    }
    return;
  } finally {
    if (settings.debug.memory === true) {
      const memDepois = process.memoryUsage();
      console.log("Sharp Compress Images");
      const tableData = [
        ["Tipo de Memória", "Antes (MB)", "Depois (MB)"],
        ["RSS", convertSize(memAntes.rss), convertSize(memDepois.rss)],
        [
          "Heap Total",
          convertSize(memAntes.heapTotal),
          convertSize(memDepois.heapTotal),
        ],
        [
          "Heap Usado",
          convertSize(memAntes.heapUsed),
          convertSize(memDepois.heapUsed),
        ],
        [
          "External",
          convertSize(memAntes.external),
          convertSize(memDepois.external),
        ],
      ];
      console.log(table(tableData));
    }
  }
}
