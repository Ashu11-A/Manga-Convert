import axios from "axios";
import { Request, Response, NextFunction } from "express";
import { copyHeaders } from "./copyHeaders";
import { redirect } from "./redirects";
import { compress } from "../compress/compress";
import https from "https";
import { pick } from "lodash";
import { removeBackground } from "@/tensorflow/processImg";

export async function proxy(req: Request, res: Response) {
  try {
    console.log(`Imagem recebida ${req.params.url}`);
    const response = await axios.get(req.params.url, {
      ...pick(req.headers, ["cookie", "dnt", "referer"]),
      headers: {
        "user-agent": "Manga Convert by Ashu11-a",
        "x-forwarded-for": req.headers["x-forwarded-for"] || req.ip,
        via: "1.0 - beta Manga Convert",
      },
      timeout: 10000,
      maxRedirects: 5,
      responseType: "arraybuffer",
      httpsAgent: new https.Agent({ rejectUnauthorized: false }),
    });

    const buffer = Buffer.from(response.data);
    copyHeaders(response, res);
    res.setHeader("content-encoding", "identity");
    req.params.originType = response.headers["content-type"] || "";
    req.params.originSize = buffer.length.toString();

    const imageProcessed = await removeBackground(buffer)
    req.params.tensorflowSize = imageProcessed?.length.toString() ?? '0';

    await compress(req, res, imageProcessed ?? buffer);
  } catch (err) {
    console.log(err);
    return redirect(req, res);
  }
}
