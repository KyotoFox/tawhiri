# -------------------
# The build container
# -------------------
FROM debian:bookworm-slim

EXPOSE 8000/tcp

RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-cffi \
    libffi-dev \
    python3-wheel \
    unzip \
    imagemagick \
    tini \
    proj-bin

#  rm -rf /var/lib/apt/lists/*

COPY requirements.txt /root/tawhiri/

WORKDIR /root/tawhiri
RUN pip3 install --user --no-warn-script-location --ignore-installed --break-system-packages -r requirements.txt

COPY . /root/tawhiri

WORKDIR /root/tawhiri/magicmemoryview
RUN python3 setup.py build && python3 setup.py install
WORKDIR /root/tawhiri

# Install ourselves
RUN pip3 install --user --no-warn-script-location --break-system-packages -e .
#RUN pip3 install --user --no-warn-script-location --break-system-packages ./magicmemoryview --force-reinstall

RUN cd /root/tawhiri && \
  python3 setup.py build_ext --inplace

RUN rm /etc/ImageMagick-6/policy.xml && \
  mkdir -p /run/tawhiri

WORKDIR /root

ENV PATH=/root/.local/bin:$PATH

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD /root/.local/bin/gunicorn -b 0.0.0.0:8000 --worker-class gevent -w 1 tawhiri.api:app
