import { Response, Request } from 'express';
import settings from '@/settings.json'

export async function bypass(res: Response, buffer: Buffer): Promise<void> {
  try {
    res.setHeader('x-proxy-bypass', '1');
    res.setHeader('content-length', buffer.length.toString());
    res.status(200);
    res.write(buffer);
    res.end();
  } catch (err) {
    if (settings.debug.error === true) {
      console.log("Erro no Bypass")
      console.log(err)
    }
  }
}
