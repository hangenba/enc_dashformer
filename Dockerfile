FROM ubuntu:22.04


RUN apt-get update && \
    apt-get install -y \
    wget \
    vim \
    tar \
    openssl \ 
    && rm -rf /var/lib/apt/lists/*


RUN wget https://go.dev/dl/go1.22.6.linux-amd64.tar.gz && \
    rm -rf /usr/local/go && \
    tar -C /usr/local -xzf go1.22.6.linux-amd64.tar.gz && \
    rm go1.22.6.linux-amd64.tar.gz


ENV PATH="/usr/local/go/bin:${PATH}"

ADD ./dashformer /home


WORKDIR /home
