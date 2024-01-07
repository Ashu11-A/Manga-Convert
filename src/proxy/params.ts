import { NextFunction, Request, Response } from "express";
import settings from '@/settings.json'

interface RequestMod extends Request {
    params: Record<string, any>
}

export function params (req: RequestMod, res: Response, next: NextFunction) {
    let url = req.query.url

    console.log(url)

    if (Array.isArray(url)) url = url.join('&url=')

    if (!url || url === undefined) {
        return res.send('bandwidth-hero-proxy').status(404)
    }

    if (!Array.isArray(url) && typeof url === 'string') {
        url = url.replace(/http:\/\/1\.1\.\d\.\d\/bmi\/(https?:\/\/)?/i, 'http://')

        req.params.url = url
        req.params.webp = !req.query.jpeg;
        req.params.grayscale = req.query.bw !== '0';
        req.params.quality = parseInt(String(req.query.l), 10) || settings.default.quality

        console.log(req.params)

        next()
    }
}