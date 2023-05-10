# use a Google maintained base image hosted in 
# Google's container registry
FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-0

# package dependencies
ARG AIF_PIP_INDEX
RUN pip install -i $AIF_PIP_INDEX --upgrade pip
COPY requirements.txt
RUN pip install -i $AIF_PIP_INDEX -r requirements.txt

# copy all necessary code
COPY ./src/run.py
COPY ./src/models/w2v2_models.py
COPY ./src/utilities/dataloader_utils.py
COPY labels.txt

# execute the code
ENTRYPOINT ["python", "/run.py"]