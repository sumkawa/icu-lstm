#!/bin/bash
conda init
source ~/.bashrc  # or ~/.zshrc depending on your shell
conda activate icu_classifier_env
pip install -r requirements.txt
