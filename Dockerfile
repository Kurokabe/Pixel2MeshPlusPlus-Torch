FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu117-py38-torch201:biweekly.202310.3

RUN rm -rf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcudadebugger.so.1


# Install pip dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# COPY external/chamfer ./external/chamfer
# COPY external/neural_renderer ./external/neural_renderer

# RUN cd external/chamfer && python setup.py install
# RUN cd external/neural_renderer && python setup.py install

# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20230419.v1 /artifacts /var/
RUN /var/requirements/install_system_requirements.sh && \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default
ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888

# support Deepspeed launcher requirement of passwordless ssh login
RUN apt-get update
RUN apt-get install -y openssh-server openssh-client
