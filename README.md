<div align="center">

# [ML] Manga (Dark Mode)

![license-info](https://img.shields.io/github/license/Ashu11-A/Manga-Converter?logo=gnu&style=for-the-badge&colorA=302D41&colorB=f9e2af&logoColor=f9e2af)
![stars-infoa](https://img.shields.io/github/stars/Ashu11-A/Manga-Converter?colorA=302D41&colorB=f9e2af&style=for-the-badge)

![Last-Comitt](https://img.shields.io/github/last-commit/Ashu11-A/Manga-Converter?style=for-the-badge&colorA=302D41&colorB=b4befe)
![Comitts Year](https://img.shields.io/github/commit-activity/y/Ashu11-A/Manga-Converter?style=for-the-badge&colorA=302D41&colorB=f9e2af&logoColor=f9e2af)
![reposize-info](https://img.shields.io/github/repo-size/Ashu11-A/Manga-Converter?style=for-the-badge&colorA=302D41&colorB=90dceb)

</div>
<div align="left">




## 📃 | Description

Um simples projeto feito em Python (training) e TypeScript (proxy/tests) para remover o background de mangas. Fiz isso, pois leio mangas majoritariamente a noite.

| Input                      | Output                       |
| -------------------------- | ---------------------------- |
| ![Input](./source/input.png) | ![Output](./source/output.png) |

Esse projeto usa U-Net, e foi implementado usando Tensorflow.

U-Net article:

```
Ronneberger, Olaf, Philipp Fischer, and Thomas Brox.
"U-net: Convolutional networks for biomedical image segmentation."
In International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 234-241. Springer, Cham, 2015.
```

## ⚙️ | Requirements

| Program | Vesion   |
| ------- | -------- |
| Nodejs  | v21.5.0  |
| Python  | v3.10.12 |

## 💹 | Production (only proxy)

```sh
# Install fnm
apt install -y curl unzip

curl -fsSL https://fnm.vercel.app/install | bash
# rode o retorno do fnm, no meu caso:
# export PATH="/home/ashu/.local/share/fnm:$PATH"
eval "$(fnm env --use-on-cd)"
fnm install
fnm use

npm install

# For ARM64
npm rebuild @tensorflow/tfjs-node --build-from-source

# Start
npm run production
```

## 🐛 | Develop (training)

### Install requirements

```sh
# Windowns WSL2: https://www.tensorflow.org/install/pip?hl=pt-br#windows-wsl2_1
# Install cuda: https://developer.nvidia.com/cuda-downloads

sudo apt install nvidia-cuda-toolkit
sudo apt install -y python3.10-venv libjpeg-dev zlib1g-dev
python3.10 -m venv ./
source bin/activate

pip install -r requirements.txt
python3 -m pip install --upgrade pip setuptools wheel
sudo pip3 install pillow --no-binary :all:
```

### Training

```sh
# Active venv
source bin/activate

# Look for the best result.
python training/start.py --best

# Run a ready-made script.
python training/start.py --unet
```

##### Saving current Libs

```sh
pip freeze > .\requirements.txt
```

### Soluções de Erros

##### Error code: Wsl/Service/CreateInstance/MountVhd/HCS/ERROR_FILE_NOT_FOUND

Causa: Possivelmente você desistalou e reinstalou o wsl/distro.

Solução:

```
# List the distributions installed, by running following in PowerShell.
wsl -l

# Unregister the distribution. Replace the "Ubuntu" below with your distribution name found in Step #1:
wsl --unregister Ubuntu-22.04

# Launch the Ubuntu (or other distribution) which was installed using Microsoft Store
```
