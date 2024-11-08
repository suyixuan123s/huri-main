pip download -r requirements.txt -d ../package
pip install --no-index --find-links=../package -r requirements.txt
