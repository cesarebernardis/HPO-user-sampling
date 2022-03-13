FROM ubuntu:20.04

# copy source code
ADD requirements.txt /.

# update libs
RUN apt-get update
RUN apt-get install -y

# install python
RUN apt-get install -y build-essential python3.6
RUN apt-get install -y build-essential python3-pip

# install dependencies
RUN pip install Cython
RUN pip install -r requirements.txt

# cython
# RUN python3 /thesis/setup.py build_ext --inplace
# RUN python3 /thesis/setup.py install

# run
CMD ["/bin/bash"]

# cython version
# aggiorna wheel a 0.35 ?
# tutto in python3 ?