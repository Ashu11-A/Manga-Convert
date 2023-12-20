import express from 'express'
import proxy from 'express-http-proxy'
const app = express()

app.get('/', proxy('https://www.google.com.br/'))

app.listen(3000, () => {
    console.log('Estou na rodando em: http://localhost:3000')
})
