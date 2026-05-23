# AWS Lambda container image for NutriVision (Flask + PyTorch).
#
# The 829 MB model checkpoint is downloaded from the public HF model repo
# DURING the docker build, so it ends up baked into the image. That way
# Lambda cold starts only have to mmap the image — no runtime download.
#
# Result: ~3-4 GB image (well under the 10 GB Lambda limit).
# Cold start: 20-40 s. Warm: 2-5 s.

FROM public.ecr.aws/lambda/python:3.10

# System libs that PIL / OpenCV-style ops + the CLIP `git+` install need
RUN yum install -y --setopt=tsflags=nodocs \
        mesa-libGL \
        git \
    && yum clean all && rm -rf /var/cache/yum

# Install Python deps into the Lambda task root (Lambda searches there first)
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt \
        --target ${LAMBDA_TASK_ROOT}

# Bake the trained checkpoint INTO the image — fetched from the public HF
# model repo (no auth needed). This happens once at build time, so cold
# starts skip the network roundtrip entirely.
RUN cd ${LAMBDA_TASK_ROOT} && \
    python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='aman24012/nutrivision-om-igsm', \
                filename='best_model.pt', \
                local_dir='.', \
                local_dir_use_symlinks=False)"

# App code + assets
COPY app.py lambda_handler.py     ${LAMBDA_TASK_ROOT}/
COPY models/                       ${LAMBDA_TASK_ROOT}/models/
COPY docs/                         ${LAMBDA_TASK_ROOT}/docs/

# Lambda hands incoming events to <module>.<handler>
CMD ["lambda_handler.handler"]
