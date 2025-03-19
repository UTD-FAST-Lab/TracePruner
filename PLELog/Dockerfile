# Use the official Miniconda3 image instead of Python
FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /app/PLELog

# Create a new Conda environment with Python 3.8
RUN conda create -n myenv python=3.8 -y 

# Activate the environment and install hdbscan
RUN /bin/bash -c "source activate myenv && conda install -c conda-forge hdbscan=0.8.29 -y && pip install torch overrides tqdm regex drain3"

VOLUME ["/app/PLELog"]


# Set the default shell to use the Conda environment
ENV CONDA_DEFAULT_ENV=myenv
ENV PATH="/opt/conda/envs/myenv/bin:$PATH"

# Keep the container running interactively
# CMD ["bash", "-c", "python approaches/PLELog.py --dataset BGL"]
CMD ["/bin/bash"]

