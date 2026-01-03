# Amazing International Airlines Inc. - Customer Segmentation Project

## Project Overview
This project, developed by **Group 92**, acts as a consultancy engagement for **Amazing International Airlines Inc. (AIAI)**. Our objective is to develop a data-driven customer segmentation strategy to identify distinct customer groups. By leveraging data mining techniques on customer and flight data, we provide actionable insights to enable personalized marketing and service improvements.

The analysis focuses on two primary dimensions:
1.  **Value-Based Segmentation:** Grouping customers based on economic contribution (CLV, Income, Points).
2.  **Demographic Segmentation:** Categorizing customers by personal attributes (Age, Education, Location).
3.  **Strategic Merging:** Combining these perspectives to create robust, actionable marketing personas.

## Team Members
| Name | Student ID | Email |
| :--- | :--- | :--- |
| **Mehmet Karaca** | 20250344 | 20250344@novaims.unl.pt |
| **Duarte Gomes** | 20250017 | 20250017@novaims.unl.pt |
| **Esra Salhi** | 20250537 | 20250537@novaims.unl.pt |


## Github Link
https://github.com/Mehmet1700/DataMining2025

## Repository Structure

The project is organized as follows:

### Final Submission (`submission2/`)
This folder contains the finalized, polished code used for the project deliverables.
*   **`00_Data_processing_Feature_Engineering.ipynb`**: 
    *   Data cleaning (handling missing values, duplicates, logic checks).
    *   Feature engineering (Tenure, Flight Metrics, Seasonality).
    *   Data preparation for clustering (Encoding, Scaling).
*   **`01_Clustering.ipynb`**: 
    *   Exploratory Data Analysis (EDA).
    *   Value-Based Clustering (K-Means).
    *   Demographic Clustering (K-Means).
    *   **Merged Segmentation:** Cross-tabulation of value and demographic clusters.
    *   **Strategic Analysis:** Bubble charts, geospatial mapping, and business recommendations.

### Data (`data/`)
*   **`raw/`**: Original datasets (`DM_AIAI_CustomerDB.csv`, `DM_AIAI_FlightsDB.csv`).
*   **`cleanAndFeatureEngineered/`**: Processed datasets ready for modeling.
*   **`outputs/`**: Final results, including `customers_with_clusters.csv`.

### Archive
*   `notebooks_deliverable1/`: Initial exploration and Phase 1 deliverables.
*   `notebooks_deliverable2/`: Development notebooks and intermediate steps.

## Installation & Requirements

This project requires **Python 3.x** and the following libraries:
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn`
*   `plotly`
*   `geopandas` (for geospatial visualizations)

To install the dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage Instructions

1.  **Setup:** Ensure the raw data files are placed in `data/raw/`.
2.  **Preprocessing:** Run `submission2/00_Data_processing_Feature_Engineering.ipynb` to clean the data and generate the feature-engineered datasets.
3.  **Analysis:** Run `submission2/01_Clustering.ipynb` to perform the segmentation analysis and generate visualizations.

## License
This project is part of the **Data Mining 2025** course at NOVA IMS.