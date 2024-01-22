#!/bin/bash

export ARCH=$(uname -m)
export LD_PRELOAD=/usr/lib/$ARCH-linux-gnu/libjemalloc.so.2
if [ ! -z "$(command -v nvm)" ]; then
    source "$HOME/.nvm/nvm.sh"
    export NVM_DIR="$HOME/.nvm"
    echo "Usando o NVM"
    nvm install 21
    nvm use 21
elif [ ! -z "$(command -v fnm)" ]; then
    export PATH="$HOME/.local/share/fnm:$PATH"
    echo "Usando o FNM"
    fnm install 21
    fnm use 21
elif [[ "$(node -v)" == v21.* ]]; then
    echo "Usando o Nodejs $(node -v)"
fi
node --version
npm install
if [ "$ARCH" == "aarch64"]; then
    npm rebuild @tensorflow/tfjs-node --build-from-source
fi
npm run production