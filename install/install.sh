echo "installing dependencies"
pip install -r 1_requirements.txt
pip install -r 2_requirements.txt --no-cache
pip install -r 3_requirements.txt
pip install -r dev_requirements.txt