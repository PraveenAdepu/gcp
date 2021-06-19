# For more information, please refer to https://aka.ms/vscode-docker-python
FROM apache/beam_python3.7_sdk:latest

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
#COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
# CMD ["python", "main.py"]

# docker build --tag python-docker .
# docker images
# docker run -p 5000:5000 -t -i python-docker
# docker run --publish 5000:5000 python-docker
# curl localhost:5000

# docker run -d -p 5000:5000 python-docker
# curl localhost:5000
# docker save python-docker > C:\Users\61433\Documents\statscope\python-docker\docker-python.tar
# docker load < C:\Users\61433\Documents\statscope\python-docker\docker-python.tar
