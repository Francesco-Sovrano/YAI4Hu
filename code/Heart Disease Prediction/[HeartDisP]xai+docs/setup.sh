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
