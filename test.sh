git submodule update --init --recursive && \
cd TTS && \
git fetch --tags && \
git checkout 0.1.1 && \
echo "Installing TTS..." && \
pip install --use-deprecated=legacy-resolver -e . -q && \
cd .. && \
echo "Installing other requirements..." && \
pip install -r requirements.txt -q && \
echo "Downloading Japanese/Chinese tokenizer..." && \
python -m unidic download
