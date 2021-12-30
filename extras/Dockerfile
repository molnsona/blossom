FROM debian:testing

RUN echo 'deb http://deb.debian.org/debian testing main contrib non-free' > /etc/apt/sources.list && apt-get update && apt-get install -y build-essential git cmake libsdl2-dev nvidia-cuda-toolkit && rm -fr /var/*/apt
