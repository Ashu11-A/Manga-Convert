import express, { Router } from "express";
import { params, proxy } from "./proxy";

const router: Router = Router()
router.get('/img', params, proxy)

router.use('/public', express.static(__dirname + '/public'));

export { router }