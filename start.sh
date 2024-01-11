ARCH=$(uname -m)

source "/home/container/.nvm/nvm.sh"
export NVM_DIR=/home/container/.nvm
export LD_PRELOAD=/usr/lib/$ARCH-linux-gnu/libjemalloc.so.2
nvm install 21
nvm use 21
npm install
npm rebuild @tensorflow/tfjs-node --build-from-source
npm run production