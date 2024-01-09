import { AxiosResponse } from 'axios'
import { ServerResponse } from 'http'

export function copyHeaders(origin: AxiosResponse<any, any>, destination: ServerResponse): void {
  if (!origin.headers || typeof origin.headers !== 'object') {
    console.log('Headers inválidos ou ausentes na origem.')
    return
  }

  const headers: [string, any][] = Object.entries(origin.headers)
  for (const [headerKey, headerValue] of headers) {
    destination.setHeader(headerKey, headerValue)
  }
}
