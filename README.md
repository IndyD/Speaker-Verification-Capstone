# Speaker-Verification-Capstone
Real-time text-independent one-shot speaker verification. Capstone project for SMU MSDS graduate program.

## Maneframe Environment Setup:
#### Create a virtual env to install soundfile (it can't be installed with conda)
module load python/3
python3 -m venv ~/.venv/capstone_env
source ~/.venv/capstone_env/bin/activate
pip3 install --upgrade pip
pip install soundfile
#### Also install librosa for spectral processing
pip install soundfile

#### Insatll ffmpeg to convert audio 
conda install -c conda-forge ffmpeg
