# Use Ubuntu as the base image
FROM ubuntu:latest

RUN apt-get update \
    && apt-get install -y htop nano python3-dev python3-pip libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev

# Set the working directory to /app
WORKDIR /app

# # Copy the current directory contents into the container at /app
COPY . /app


# # Upgrade pip
RUN pip3 install --upgrade pip



# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port/ 8501 available to the world outside this container
EXPOSE 8501

# # Run your Streamlit app when the container launches
CMD ["streamlit", "run", "./Week_4/streamlit_demo.py"]
