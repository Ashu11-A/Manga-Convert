import sharp from "sharp";
import { Request, Response } from "express";
import { redirect } from "../proxy/redirects";
import { shouldCompress } from "./shouldCompress";
import { bypass } from "@/proxy/bypass";
import { table } from "table";
import { convertSize } from "@/functions/formatBytes";

sharp.cache(false);
sharp.concurrency(4);
sharp.simd(true);

export async function compress(
  req: Request,
  res: Response,
  input: Buffer
): Promise<void> {
  const memAntes = process.memoryUsage();
  const format = req.params.webp ? "png" : "jpeg";

  console.log("Iniciando compressão...");

  try {
    if (shouldCompress(req)) {
      const output = await sharp(input)
        .grayscale(req.params.grayscale ? true : false)
        .png({
          quality: parseInt(req.params.quality) || 80,
        })
        .toBuffer();

      const originalSize = parseInt(req.params.originSize);
      const compressedSize = output.length;
      const infoTable = [
        ["Input", "Tensor", "Output"],
        [
          convertSize(originalSize),
          convertSize(parseInt(req.params.tensorflowSize)),
          convertSize(compressedSize),
        ],
      ];
      console.log(table(infoTable));

      if (compressedSize >= originalSize) {
        console.log(
          `A compressão resultou em um tamanho de arquivo maior(${compressedSize}). Retornando a imagem original.`
        );

        res.setHeader("content-type", `image/${format}`);
        res.setHeader("content-length", originalSize.toString());
        res.setHeader("x-original-size", originalSize.toString());
        res.status(200);
        res.send(input);
      } else {
        console.log("Imagem comprimida com sucesso.");
        res.setHeader("content-type", `image/${format}`);
        res.setHeader("content-length", compressedSize.toString());
        res.setHeader("x-original-size", originalSize.toString());
        res.setHeader(
          "x-bytes-saved",
          (originalSize - compressedSize).toString()
        );
        res.status(200);
        res.send(output);
      }
      output.fill(0);
    } else {
      await bypass(req, res, input);
    }
  } catch (err) {
    return redirect(req, res);
  } finally {
    input.fill(0)
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
    // if (global.gc) {
    //   const memAntes = process.memoryUsage();
    //   console.log("Solicitando ao gc para liberar a memória não utilizada");
    //   global.gc();
    //   const memDepois = process.memoryUsage();

    //   const tableData = [
    //     ["Tipo de Memória", "Antes (MB)", "Depois (MB)"],
    //     ["RSS", convertSize(memAntes.rss), convertSize(memDepois.rss)],
    //     [
    //       "Heap Total",
    //       convertSize(memAntes.heapTotal),
    //       convertSize(memDepois.heapTotal),
    //     ],
    //     [
    //       "Heap Usado",
    //       convertSize(memAntes.heapUsed),
    //       convertSize(memDepois.heapUsed),
    //     ],
    //     [
    //       "External",
    //       convertSize(memAntes.external),
    //       convertSize(memDepois.external),
    //     ],
    //   ];
    //   console.log(table(tableData));
    // }
  }
}
