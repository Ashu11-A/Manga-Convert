import { Request, Response } from 'express';

export function redirect(req: Request, res: Response): void {
  console.log(`Pulando: ${req.params.url}`)
  if (res.headersSent) return;

  res.setHeader('content-length', '0');
  res.removeHeader('cache-control');
  res.removeHeader('expires');
  res.removeHeader('date');
  res.removeHeader('etag');
  res.setHeader('location', encodeURI(String(req.params.url)));
  res.status(302).end();
}
