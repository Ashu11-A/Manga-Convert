import express, { Application } from "express";
import { router } from "@/router";

export class App{
    public server: Application

    constructor(){
        this.server = express()
        this.middleware()
        this.router()
    }

    private middleware() {
        this.server.use(express.json())
    }

    private router(){
        this.server.use(router)
    }
}