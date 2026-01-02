# Data Mining 2025 - Customer Segmentation (Group 92)

## Project Overview
This project aims to perform comprehensive customer segmentation for an airline company using data mining techniques. By analyzing customer and flight data, we identify distinct customer groups based on value, behavior, and demographics to support targeted marketing strategies and operational improvements.

## Team Members
- **Mehmet**
- **Esra**
- **Duarte**

## Project Structure

The repository is organized as follows:

### 📂 Data
- **`data/raw/`**: Original datasets (`DM_AIAI_CustomerDB.csv`, `DM_AIAI_FlightsDB.csv`, `DM_AIAI_Metadata.csv`).
- **`data/cleaned/`**: Cleaned versions of the datasets.
- **`data/aggregated/`**: Aggregated customer data.
- **`data/cleanAndFeatureEngineered/`**: Final datasets with feature engineering applied, ready for modeling.

### 📓 Notebooks
The analysis is divided into deliverables:

#### Deliverable 1 (`notebooks_deliverable1/`)
- Initial Data Overview & EDA.
- Preliminary segmentation approaches (Value-based, Behavioral, Demographic).

#### Deliverable 2 (`notebooks_deliverable2/`)
- **00_Data_processing_Feature_Engineering.ipynb**: Advanced data preprocessing and feature engineering pipeline.
- **01_Clustering_Value_Based_updated.ipynb**: Updated value-based clustering (e.g., RFM analysis).
- **02_Clustering_Behavioral_updated.ipynb**: Updated behavioral segmentation.
- **03_Clustering_Demographic_updated.ipynb**: Updated demographic segmentation.
- **04_Clustering_merged_perspectives.ipynb**: Integration of different segmentation perspectives.
- **05_Clustering_Visualization.ipynb**: Visualizations of the final clusters.


The Final submissions and their notebooks are inside submission1 and submission2 folders

## Installation & Requirements

This project requires Python 3.x and the following libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- plotly
- joblib

To install the dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
1. Ensure the data files are placed in `data/raw/`.
2. Run the notebooks in `notebooks_deliverable2/` in numerical order to reproduce the analysis pipeline.

## License
This project is part of the Data Mining 2025 course.