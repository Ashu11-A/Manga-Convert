# Manga-Converter

### Requisitos

- Nodejs v18.13.0
- Python 3.7.9

### Instalação

##### Requirements

```sh
pip install -r requirements.txt
```

##### Pillow

```sh
apt install libjpeg-dev zlib1g-dev

python3 -m pip install --upgrade pip setuptools wheel
sudo pip3 install pillow --no-binary :all:
```

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

### Treinamento

```sh
# Procure o melhor resultado.
python training/training.py --best

# Rode um script já feito.
python training/training.py
```

##### Salvar Libs atuais
```sh
pip freeze > .\requirements.txt
```

### Possíveis Erros

#### Windows
- Comando de inicialização do ambiente virtual dá erro:

```sh
O arquivo Scripts\Activate.ps1 não pode ser
carregado porque a execução de scripts foi desabilitada
neste sistema. Para obter mais informações, consulte
about_Execution_Policies em
https://go.microsoft.com/fwlink/?LinkID=135170.
```
- Solução: Rode o PowerShell em modo Administrativo, e use o comando
```sh
 Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```