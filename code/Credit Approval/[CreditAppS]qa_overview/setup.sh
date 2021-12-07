#!/bin/bash

MY_DIR="`python -c "import os; print(os.path.realpath('$1'))"`"
cd $MY_DIR

# Run AIX server
cd aix
echo 'Setting up AIX server..'
virtualenv .env -p python3.7
source .env/bin/activate
pip install pip==21.1.1
pip install --use-deprecated=legacy-resolver -U setuptools wheel twine
pip install  --use-deprecated=legacy-resolver -r requirements.txt
cd ..

# Run YAI server
cd yai
echo 'Setting up YAI server..'
virtualenv .env -p python3.7
source .env/bin/activate
pip install pip==21.1.1
pip install --use-deprecated=legacy-resolver -U setuptools wheel twine
pip install  --use-deprecated=legacy-resolver -r requirements.txt
cd ..

# Run OKE Server
cd oke
echo 'Setting up OKE server..'
virtualenv .env -p python3.7
source .env/bin/activate
pip install pip==21.1.1
pip install --use-deprecated=legacy-resolver -U setuptools wheel twine
echo 'Install QuAnsX'
pip install  --use-deprecated=legacy-resolver -e /home/toor/Desktop/packages/quansx
echo 'Install KnowPy'
pip install  --use-deprecated=legacy-resolver -e /home/toor/Desktop/packages/knowpy
# cd .env/lib
# git clone https://github.com/huggingface/neuralcoref.git
# cd neuralcoref
# pip install  --use-deprecated=legacy-resolver -r requirements.txt
# pip install  --use-deprecated=legacy-resolver -e .
# cd ..
# cd ../..
pip install  --use-deprecated=legacy-resolver -r requirements.txt
python3 -m spacy download en_core_web_trf
# python3 -m spacy download en_core_web_sm
python3 -m nltk.downloader stopwords punkt averaged_perceptron_tagger framenet_v17 wordnet brown
cd ..
