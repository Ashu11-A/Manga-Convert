import { Response } from "express";

export async function send(res: Response, img: Buffer, format: string  ) {
  res.setHeader("content-type", `image/${format}`);
  res.setHeader("content-length", img.length.toString());
  res.status(200);
  res.send(img);
}
