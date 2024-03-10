import axios from "axios";
import { Request, Response } from "express";
import { copyHeaders } from "./copyHeaders";
import { redirect } from "./redirects";
import { compress } from "../compress/compress";
import https from "https";
import { pick } from "lodash";
import { removeBackground } from "@/tensorflow/processImg";
import sharp from "sharp";
import settings from "@/settings.json";
import probe from "probe-image-size";
import { shouldCompress } from "@/compress/shouldCompress";
import { send } from "./send";

export async function proxy(req: Request, res: Response) {
  const { width, height, length } = await probe(req.params.url, {
    rejectUnauthorized: false,
  }).catch(() => {
    return { width: 1, height: 1, length: 0 };
  });
  const range = width / height;
  if (range <= 0.5 || range >= 0.8) {
    console.log(
      `Imagem ${
        range === 1
          ? "Invalida"
          : range <= 0.5
          ? "muito pequena"
          : "muito grande"
      }`
    );
    console.log(range, width, height)
    return redirect(req, res);
  }

  if (length >= 1024 * 1024) {
    // 10Mb
    console.log(`Imagem excede o limite de 10Mb!`);
    return redirect(req, res);
  }

  await axios
    .get(req.params.url, {
      ...pick(req.headers, ["cookie", "dnt", "referer"]),
      headers: {
        "user-agent": "Manga Convert by: Ashu11-a",
        "x-forwarded-for": req.headers["x-forwarded-for"] || req.ip,
        via: "1.0 - Manga Convert",
      },
      timeout: 30000,
      maxRedirects: 5,
      responseType: "arraybuffer",
      httpsAgent: new https.Agent({ rejectUnauthorized: false }),
    })
    .then(async (response) => {
      try {
        console.log(`Request: ${req.params.url}`);
        const format = req.params.png ? "png" : "jpeg";
        const buffer = await sharp(Buffer.from(response.data)).png().toBuffer();

        copyHeaders(response, res);
        res.setHeader("content-encoding", "identity");
        req.params.originType = response.headers["content-type"] || "";
        req.params.originSize = buffer.length.toString();

        const imageProcessed = await removeBackground(buffer);
        req.params.tensorflowSize = imageProcessed?.length.toString() ?? "0";

        if (shouldCompress(req)) {
          const imageCompress = await compress(
            req,
            res,
            imageProcessed ?? buffer
          );
          if (imageCompress !== undefined) {
            await send(res, imageCompress, format);
          }
        } else {
          if (imageProcessed === undefined) return redirect(req, res);
          await send(res, imageProcessed, format);
        }
      } catch (err) {
        if (settings.debug.error === true) {
          console.log("Erro no Proxy");
          console.log(err);
        }
        return redirect(req, res);
      }
    })
    .catch((err) => {
      if (settings.debug.request === true) {
        console.log(`Erro na requicição: ${req.params.url}`);
        console.log(err);
      }
      return redirect(req, res);
    });
}
