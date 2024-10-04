# Machine Learning Housing Corporation

## Overview

Welcome to the Machine Learning Housing Corporation! This project aims to use California census data to build a model for predicting housing prices in the state. The data includes metrics such as population, median income, and median housing price for each block group, referred to as "districts."

## Project Goal

Our goal is to create a model that predicts the median housing price in any district, given various metrics. This prediction will be used by another machine learning system to determine investment opportunities, which is crucial for revenue generation.

## Strategy

### Task Definition
- Supervision Type: Supervised learning
- Problem Type: Regression (multiple regression, univariate regression)
- Learning Technique: Batch learning

### Performance Measure
- Metric: Root Mean Square Error (RMSE)

### Assumptions
- The predicted prices will be directly used by a downstream machine learning system.
- Confirmed with the downstream team that actual prices, not categories, are needed.

## Data Description

The dataset includes the following attributes:

| Column Name          | Data Type | Description                                                                                     |
|----------------------|-----------|-------------------------------------------------------------------------------------------------|
| `longitude`          | float64   | Distance from the Prime Meridian (negative values, e.g., -110.35)                               |
| `latitude`           | float64   | Distance from the Equator (values e.g., 30.2, 37.88)                                            |
| `housing_median_age` | float64   | Age of houses (capped at 50 years)                                                              |
| `total_rooms`        | float64   | Number of rooms in all houses in a district                                                     |
| `total_bedrooms`     | float64   | Number of bedrooms in all houses in a district                                                  |
| `population`         | float64   | Population of each district (ranges from 600 to about 3000 people)                              |
| `households`         | float64   | Number of families in each district                                                             |
| `median_income`      | float64   | Scaled income (0.5 to 15, each increment represents $10,000)                                    |
| `median_house_value` | float64   | Median house value (capped at $500,000)                                                         |
| `ocean_proximity`    | object    | Proximity to the ocean (values: <1H OCEAN, INLAND, NEAR OCEAN, NEAR BAY, ISLAND)                |

## Quick Start

### Get the Dataset

To load the dataset, use the following function in your Jupyter Notebook:

```python
# Function to load the dataset
def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    csv_path = Path("datasets/housing/housing.csv")
    
    # Check if the CSV file already exists
    if csv_path.is_file():
        print(f"Loading data from {csv_path}")
        return pd.read_csv(csv_path)
    
    # If the CSV file doesn't exist, download and extract the data
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, tarball_path)
        print("Download complete.")
    
    # Extract the .tgz file
    with tarfile.open(tarball_path) as housing_tarball:
        print("Extracting files...")
        housing_tarball.extractall(path="datasets")
        print("Extraction complete.")
    
    # Verify the contents of the extracted directory
    extracted_files = list(Path("datasets").glob("*"))
    print(f"Extracted files: {extracted_files}")

    # Load the data
    return pd.read_csv(csv_path)()

    # Call data loading function
    housing = load_housing_data()
```

### Take a Quick Look at the Data Structure

Inspect the top five rows of the dataset:

```python
housing.head()
```

This will display the first five rows, each representing one district with the attributes listed above.

## Conclusion

With this setup, you are ready to start designing and coding your predictive model. Good luck, and let's improve the accuracy of housing price predictions in California!

---

This README provides a clear and concise overview of the project, covering the essential details and instructions for getting started. You can include images within the README using the Markdown syntax for images if needed.

## Data Exploration and Visualization

Understanding the data through various visualizations and statistical analyses.

## Data Preprocessing

Cleaning and preparing the data for modeling.

## Model Building

Selecting and training machine learning models.

## Model Evaluation

Assessing the performance of the models.

## Model Fine-Tuning

Optimizing the models for better performance.

## Model Deployment

Deploying the final model for practical use.

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
