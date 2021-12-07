#!/bin/bash

MY_DIR="`python -c "import os; print(os.path.realpath('$1'))"`"
cd $MY_DIR

# Run YAI server
cd yai
echo 'Setting up YAI server..'
virtualenv .env -p python3.7
source .env/bin/activate
pip install --use-deprecated=legacy-resolver -U pip setuptools wheel twine
pip install --use-deprecated=legacy-resolver -r requirements.txt
cd ..

# Run SHAP server
cd shap
echo 'Setting up SHAP server..'
virtualenv .env -p python3.7
source .env/bin/activate
pip install --use-deprecated=legacy-resolver -U pip setuptools wheel twine
pip install --use-deprecated=legacy-resolver -r requirements.txt
# cd model
# python3 pycaret_heart_disease.py
# cd ..
cd ..

# Run OKE Server
cd oke
echo 'Setting up OKE server..'
virtualenv .env -p python3.7
source .env/bin/activate
pip install --use-deprecated=legacy-resolver -U pip setuptools wheel twine
echo 'Install QuAnsX'
pip install  --use-deprecated=legacy-resolver -e /home/toor/Desktop/packages/quansx
echo 'Install KnowPy'
pip install  --use-deprecated=legacy-resolver -e /home/toor/Desktop/packages/knowpy
# cd .env/lib
# git clone https://github.com/huggingface/neuralcoref.git
# cd neuralcoref
# pip install --use-deprecated=legacy-resolver -r requirements.txt
# pip install --use-deprecated=legacy-resolver -e .
# cd ..
# cd ../..
pip install --use-deprecated=legacy-resolver -r requirements.txt
python3 -m spacy download en_core_web_trf
# python3 -m spacy download en_core_web_sm
python3 -m nltk.downloader stopwords punkt averaged_perceptron_tagger framenet_v17 wordnet brown
cd ..
