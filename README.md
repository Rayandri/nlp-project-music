# nlp-project-music

# instruction d'installation

# 1. initialiser l'environnement virtuel
```
python -m venv env
```
or 
```
conda create -n env python=3.8
```
or on macos brew
```
brew install pyenv
pyenv install 3.8.10
pyenv virtualenv 3.8.10 env
pyenv activate env
```
# 2. activer l'environnement virtuel
```
source env/bin/activate
```
or 
```
conda activate env
```
or on macos
```
pyenv activate env
```
# 3. installer les dépendances
```
pip install -r requirements.txt
```
or 
```
conda install --file requirements.txt
```
or on macos
```
brew install -r requirements.txt
```
# 4. lancer le projet
```
python main.py
```
