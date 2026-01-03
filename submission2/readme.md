# Final Submission - Group 92

## Project Repository
**GitHub Link:** [https://github.com/Mehmet1700/DataMining2025](https://github.com/Mehmet1700/DataMining2025)

## Overview
This folder (`submission2`) contains the finalized and polished deliverables for the Amazing International Airlines Inc. customer segmentation project. These notebooks represent the complete analytical pipeline, from raw data processing to strategic business recommendations.

## Contents

### 1. Data Processing & Feature Engineering
*   **File:** `00_Data_processing_Feature_Engineering.ipynb`
*   **Description:** This notebook handles the data preparation phase.
    *   **Data Cleaning:** Addresses missing values, duplicates, and logical inconsistencies in the raw data.
    *   **Feature Engineering:** Creates new variables such as `CustomerTenureDays`, aggregated flight metrics (e.g., `total_distance`, `points_redemption_ratio`), and seasonality indicators.
    *   **Preprocessing:** Performs One-Hot Encoding for categorical variables and scaling for numerical variables to prepare the dataset for clustering.
    *   **Output:** Generates clean CSV files in `data/cleanAndFeatureEngineered/`.

### 2. Clustering & Strategic Analysis
*   **File:** `01_Clustering.ipynb`
*   **Description:** This notebook performs the core segmentation analysis.
    *   **Exploratory Data Analysis (EDA):** Visualizes key distributions and correlations.
    *   **Value-Based Clustering:** Segments customers based on economic value (CLV, Income) using K-Means (K=4).
    *   **Demographic Clustering:** Segments customers based on profile attributes (Education, Location) using K-Means (K=6).
    *   **Merged Segmentation:** Combines the two perspectives into a strategic matrix.
    *   **Business Insights:** Includes bubble charts for portfolio analysis and geospatial maps for regional targeting.

### 3. Data Folder
*   **Folder:** `data/`
*   **Description:** Contains the output of the analysis, including the final `customers_with_clusters.csv` which assigns each customer to their respective segment.