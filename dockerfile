# Use a base image with Python and Java pre-installed
FROM python:3.10-slim

# Install Java
RUN apt-get update && apt-get install -y openjdk-8-jdk

# Set environment variables (Optional)
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Copy your Python scripts
COPY . /app

# Set the working directory inside the container
WORKDIR /app

# Install Python dependencies (if any)
RUN pip install -r requirements.txt


#clone and install javaq and add it to the path



#clone NJR-1 dataset



# Set the default command for the container (running your script)
CMD ["python", "your_script.py"]
