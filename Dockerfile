# set base image (host OS)
FROM python:3.8

# set the working directory in the container
WORKDIR /agencia

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# copy the content of the local src directory to the working directory
COPY agencia/ .

# command to run on container start
CMD ["bash"] 

# CMD ["python", "0_project/0_LunarLander-v2.py" ]