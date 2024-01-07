import sharp from "sharp";
import { Request, Response } from "express";
import { redirect } from "../proxy/redirects";
import { shouldCompress } from "./shouldCompress";
import { bypass } from "@/proxy/bypass";

export async function compress(
  req: Request,
  res: Response,
  input: Buffer
): Promise<void> {
  const format = req.params.webp ? "webp" : "jpeg";

  console.log("Iniciando compressão...");

  try {
    if (shouldCompress(req)) {
      const output = await sharp(input)
        .grayscale(req.params.grayscale ? true : false)
        .toFormat(format, {
          quality: parseInt(req.params.quality) || 80,
          progressive: true,
          optimizeScans: true,
        })
        .toBuffer();

      const originalSize = parseInt(req.params.originSize);
      const compressedSize = output.length;

      function convertSize(sizeInBytes: number) {
        if (sizeInBytes < 1024) {
          return sizeInBytes + " B";
        } else if (sizeInBytes < 1048576) {
          return (sizeInBytes / 1024).toFixed(2) + " KB";
        } else {
          return (sizeInBytes / 1048576).toFixed(2) + " MB";
        }
      }

      console.log(
        "Input:",
        convertSize(originalSize),
        'Tensorflow',
        convertSize(parseInt(req.params.tensorflowSize)),
        "Output:",
        convertSize(compressedSize)
      );

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
    } else {
      await bypass(req, res, input);
    }
  } catch (err) {
    console.log(err);
    return redirect(req, res);
  }
}
