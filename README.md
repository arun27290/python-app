tar -xzvf packages_bundle.tar.gz

python -m venv venv

source venv/bin/activate

pip install --no-index --find-links=packages -r requirements.txt
