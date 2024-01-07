import { App } from "./app";
import settings from './settings.json'

const app = new App()

app.server.enable('trust proxy')

app.server.listen(settings.port, () => {
  console.log(`Servidor listado em http://localhost:${settings.port}`)
})