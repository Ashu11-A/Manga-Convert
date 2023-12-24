# Manga-Converter

### Requisitos

- Nodejs v18.13.0
- Python 3.7.9


### Treinamento - Develop

##### Iniciar Ambiente Virtual
WSL2: https://www.tensorflow.org/install/pip?hl=pt-br#windows-wsl2_1

###### Windowns
```sh
python.exe -m venv ./

# PowerShell
.\Scripts\Activate.ps1

# Prompt
.\Scripts\activate.bat
```

###### Linux
```sh
virtualenv ./

source bin/activate
```

##### Salvar Libs atuais
```sh
pip freeze > .\requirements.txt
```

##### Erros e suas soluções
- Comando de inicialização do ambiente virtual dá erro:

```sh
O arquivo Scripts\Activate.ps1 não pode ser
carregado porque a execução de scripts foi desabilitada
neste sistema. Para obter mais informações, consulte
about_Execution_Policies em
https://go.microsoft.com/fwlink/?LinkID=135170.
```
Solução:
Rode o PowerShell em modo Administrativo, e use o comando
```sh
 Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```