import axios from "axios";
import { Request, Response, response } from "express";
import { copyHeaders } from "./copyHeaders";
import { redirect } from "./redirects";
import { compress } from "../compress/compress";
import https from "https";
import { pick } from "lodash";
import { removeBackground } from "@/tensorflow/processImg";
import sharp from "sharp";
import sizeOf from "image-size";

sharp.cache(false);
sharp.concurrency(4);
sharp.simd(true);

export async function proxy(req: Request, res: Response) {
  console.log(`Request: ${req.params.url}`);
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
      maxContentLength: 10485760, // 10Mb
      responseType: "arraybuffer",
      httpsAgent: new https.Agent({ rejectUnauthorized: false }),
    })
    .then(async (response) => {
      try {
        const buffer = await sharp(Buffer.from(response.data)).png().toBuffer();

        const { width, height } = sizeOf(buffer);
        if (((width ?? 0) <= 128 || (height ?? 0) <= 512)) {
          console.log('Imagem muito pequena!')
          return redirect(req, res);
        }

        copyHeaders(response, res);
        res.setHeader("content-encoding", "identity");
        req.params.originType = response.headers["content-type"] || "";
        req.params.originSize = buffer.length.toString();

        const imageProcessed = await removeBackground(buffer);
        req.params.tensorflowSize = imageProcessed?.length.toString() ?? "0";

        await compress(req, res, imageProcessed ?? buffer);
      } catch {
        console.log("Erro no Proxy")
        return redirect(req, res);
      }
    })
    .catch(() => {
      console.log(`Erro na requicição: ${req.params.url}`);
      return redirect(req, res);
    });
}
