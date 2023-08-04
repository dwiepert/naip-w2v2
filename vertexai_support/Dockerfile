# use a Google maintained base image hosted in 
# Google's container registry
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-13.py310

# package dependencies - install requirements not included in base image
ARG AIF_PIP_INDEX
RUN pip install -i $AIF_PIP_INDEX --upgrade pip

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

#if you want to install with conda
#RUN conda install -y package_name=version

# copy all necessary code
#1. make src directory
RUN mkdir /src      

#2. set src as working dir
WORKDIR /src

#3. copy all files from src
COPY ./src /src

#4. set python path
RUN export PYTHONPATH=/src/

#RUN dir # for checking file list

# execute the code
#CMD ["python", "run.py"]
ENTRYPOINT ["python", "run.py"]