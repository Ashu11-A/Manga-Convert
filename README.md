<div align="center">

# Manga Converter

![license-info](https://img.shields.io/github/license/Ashu11-A/Manga-Converter?logo=gnu&style=for-the-badge&colorA=302D41&colorB=f9e2af&logoColor=f9e2af)
![stars-infoa](https://img.shields.io/github/stars/Ashu11-A/Manga-Converter?colorA=302D41&colorB=f9e2af&style=for-the-badge)

![Last-Comitt](https://img.shields.io/github/last-commit/Ashu11-A/Manga-Converter?style=for-the-badge&colorA=302D41&colorB=b4befe)
![Comitts Year](https://img.shields.io/github/commit-activity/y/Ashu11-A/Manga-Converter?style=for-the-badge&colorA=302D41&colorB=f9e2af&logoColor=f9e2af)
![reposize-info](https://img.shields.io/github/repo-size/Ashu11-A/Manga-Converter?style=for-the-badge&colorA=302D41&colorB=90dceb)

</div>
<div align="left">

## 📃 | Description

## ⚙️ | Requirements
| Program | Vesion |
|--|--|
| Nodejs | v21.5.0 |
| Python | v3.10.12 |

## 💹 | Production

```
npm install

# For ARM64
npm rebuild @tensorflow/tfjs-node --build-from-source

# Start
npm run production
```

## 🐛 | Develop


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

##### Start Virtual Environment

WSL2: https://www.tensorflow.org/install/pip?hl=pt-br#windows-wsl2_1

###### Linux

```sh
virtualenv ./

source bin/activate
```

### Training

```sh
# Look for the best result.
python training/training.py --best

# Run a ready-made script.
python training/training.py
```

##### Saving current Libs

```sh
pip freeze > .\requirements.txt
```