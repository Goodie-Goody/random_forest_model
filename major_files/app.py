from flask import Flask, request, jsonify, render_template, send_file, url_for
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from io import StringIO, BytesIO
import os

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)

default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())

app = Flask(__name__)

model = joblib.load("my_california_housing_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form['input_data']
    input_df = pd.read_csv(StringIO(input_data))
    
    expected_columns = [
        "longitude", "latitude", "housing_median_age", "total_rooms",
        "total_bedrooms", "population", "households", "median_income", "ocean_proximity"
    ]
    input_df = input_df[expected_columns]
    
    prediction = model.predict(input_df)
    
    input_df['prediction'] = prediction
    
    csv_output = input_df.to_csv(index=False)
    
    csv_filename = "predictions.csv"
    with open(csv_filename, "w", encoding='utf-8') as f:
        f.write(csv_output)
    
    download_link = url_for('download_file', filename=csv_filename)
    
    return jsonify({'prediction': prediction.tolist(), 'download_link': download_link})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    input_df = pd.read_csv(file)
    
    expected_columns = [
        "longitude", "latitude", "housing_median_age", "total_rooms",
        "total_bedrooms", "population", "households", "median_income", "ocean_proximity"
    ]
    input_df = input_df[expected_columns]
    
    prediction = model.predict(input_df)
    
    input_df['prediction'] = prediction
    
    csv_output = input_df.to_csv(index=False)
    
    csv_filename = "predictions.csv"
    with open(csv_filename, "w", encoding='utf-8') as f:
        f.write(csv_output)
    
    download_link = url_for('download_file', filename=csv_filename)
    
    return jsonify({'prediction': prediction.tolist(), 'download_link': download_link})

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, mimetype='text/csv', as_attachment=True, download_name=filename)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)