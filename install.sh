conda create -n autorecommender python=3.7
conda activate autorecommender
sudo apt install gcc
pip install numpy pandas torch==1.7.1 tensorflow==1.15.2 scipy cornac sklearn pyspark fastai surprise psutil
pushd recommenders && pip install -e . && popd
