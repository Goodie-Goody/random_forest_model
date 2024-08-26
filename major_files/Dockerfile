# Use an official Python runtime as a parent image
FROM python:3.11.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file to the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Install jupyter
RUN pip install jupyter

# Run the notebook to generate the model if needed
RUN jupyter nbconvert --to notebook --execute machine_learning_project.ipynb

# Copy the rest of the application code to the container
COPY . .

# Make port 80 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]
