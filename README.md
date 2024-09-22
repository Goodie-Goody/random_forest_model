# End-to-End Machine Learning Project

This project is based on Chapter 2 of Aurélien Géron’s book, Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. It demonstrates an end-to-end machine learning workflow, from data exploration to model deployment.

## Project Overview

This project involves building a machine learning model from scratch to predict housing prices in California using the California Housing Dataset from the 1990 census. The steps include:

## Problem Definition: Predicting housing prices based on various features from the dataset.

## Data Collection: Using the California Housing Dataset gathered from the 1990 census.

## Data Exploration and Visualization: Understanding the data through various visualizations and statistical analyses.

## Data Preprocessing: Cleaning and preparing the data for modeling.

## Model Building: Selecting and training machine learning models.

## Model Evaluation: Assessing the performance of the models.

## Model Fine-Tuning: Optimizing the models for better performance.

## Model Deployment: Deploying the final model for practical use.

## Installation
To run this project, you need to have Python and the following libraries installed:

Scikit-Learn, Pandas, NumPy, Matplotlib, Flask

You can install the required libraries using the following command:

  pip install -r requirements.txt

Usage

Clone the repository:
git clone https://github.com/yourusername/your-repo-name.git

Navigate to the project directory:
cd your-repo-name

Run the Jupyter Notebook to generate the model:
jupyter notebook notebooks/your_notebook.ipynb

Once the model is generated, build the Docker image:
docker build -t your-image-name -f major_files/Dockerfile .

Run the Docker container:
docker run -p 5000:5000 your-image-name

Open your web browser and go to:
http://localhost:5000

## Project Structure

1. major_files/: Contains the main files for the project.
2. Dockerfile: Docker configuration file.
3. requirements.txt: List of dependencies for the Flask app.
4. app.py: Main application file that launches the Flask app.
5. templates/: Contains HTML templates.
6. index.html: The main HTML file deployed by the Flask app.
7. data/: Contains the dataset used for the project.
8. notebooks/: Jupyter Notebooks with the project code and analysis.
9. models/: Directory for saving models (not included in the repository due to size).
10. README.md: Project documentation.

## Deployment
The project is deployed on Fly.io and can be accessed at https://machine-learning-projects-books.fly.dev/.
