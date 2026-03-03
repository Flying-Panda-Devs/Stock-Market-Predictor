#!/bin/bash
sudo apt update
sudo apt install python python3-pip -y
python -m venv stocks
source stocks/bin/activate
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
