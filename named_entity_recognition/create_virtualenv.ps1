python -m venv token_classification-env
./token_classification-env/Scripts/Activate.ps1
python -m pip install --upgrade pip
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r ./requirements.txt