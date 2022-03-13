conda create -y -n recsys-hpo --file requirements-conda.txt -c anaconda -c conda-forge

source activate recsys-hpo

pip install similaripy 
pip install interaction==1.3 kaleido==0.0.3.post1 similarity==0.0.1
pip install tensorflow-gpu==2.4

cd ../../../..
