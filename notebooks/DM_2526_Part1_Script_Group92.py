# %% [markdown]
# <div class="alert alert-block alert-info">
#   <center> <h1> Exploratory Data Analysis 
#   - Amazing International Airlines Inc. </h1> </center> <br>
#   <center> <h2> Group 92 </h2> </center> <br>
#   <center <h3> notebook  </h3> </center>
#   <center> <h3> 2025/2026 </h3> </center>
# </div>
# 

# %% [markdown]
# This Project was done by:
# 
# 
# Student Name    -   Mehmet Karaca;
# student id      -   20250344;
# contact email   -   20250344@novaims.unl.pt
# 
# Student Name    -   Duarte Gomes;
# student id      -   20250017;
# contact email   -   20250017@novaims.unl.pt
# 
# Student Name    -   Esra Salhi
# student id      -   20250537
# contact email   -   20250537@novaims.unl.pt

# %% [markdown]
# # Table of Contents
# 
# - 1. Context and Metadata (#context-metadata)
#   - 1.1 Importing Libraries (#importing-libraries)
#   - 1.2 Loading and Reading Data (#loading-and-reading-data)
#   - 1.3 Brief Preliminary Analysis (#brief-preliminary-analysis)
# - 2. Missing Values and Data Validity Checks (#data-validity-checks)
#   - 2.1 Missing Values (#missing-values)
#   - 2.2 Validity Checks (#validity-check)
#     - 2.2.1 Categorical Features of Customers (#categorical-features)
#     - 2.2.2 Numerical Features of Customers (#numerical-features-customers)
#     - 2.2.3 Numerical Features of Flights (#numerical-features-flights)
#     - 2.2.4 Datetime Features of Flights (#datetime-features-flights)
#   - 2.3 Logic Checks (#logic-checks)
#     - 2.3.1 Logic Check - DistanceKM and NumFlights (#distancekm-and-numflights)
#     - 2.3.2 Logic Check - NumFlights and FlightCompanions (#numflights-vs-companions)
# - 3. Feature Engineering (#feature-engineering)
#   - 3.1 Active Customers (#active-customers)
#   - 3.2 Tenure Days (#tenure-days)
#   - 3.3 Flight Metrics per Customer (#flight-metrics-per-customer)
#     - 3.3.1 Flights Aggregated Metrics (#flights-aggregated-metrics)
#     - 3.3.2 Merge Flights Aggregated with Customers (#merge-flights-aggregated-with-customers)
#   - 3.4 Logic Checks of the New Columns (#logic-checks-new-columns)
#     - 3.4.1 Logic Check - Customer Lifetime Value and Zero Flights (#clv-zero-flights)
#     - 3.4.2 Logic Check - AverageDistanceKM > 13,000 KM (#avg-distancekm-over-13000km)
#     - 3.4.3 Logic Check - Negative Customer Tenure Days (#negative-tenure-days)
# - 4. Exploratory Data Analysis (EDA) (#eda)
#   - 4.1 Customer EDA (#customers-eda)
#     - 4.1.1 Customer EDA - Categorical Features (#customers-eda-categorical)
#     - 4.1.2 Customer EDA - Numerical Features (#customers-eda-numerical)
#     - 4.1.3 Customer EDA - Multivariate Analysis (#customers-eda-multi-variate)
#   - 4.2 Flights EDA (#flights-eda)
#     - 4.2.1 Flights EDA - Numerical Features (#flights-eda-numerical)
# - 5. Segmentation Analyses (#segmentation-analyses)
#   - 5.1 Value-Based Segmentation (#value-based-segmentation)
#   - 5.2 Behavioral Segmentation (#behavioral-segmentation)
#   - 5.3 Demographic Segmentation (#demographic-segmentation)
# - 6. Conclusion (#conclusion)
# 

# %% [markdown]
# # 1. Context and Metadata <a class="anchor" id="context-metadata"></a>
# 

# %% [markdown]
# CustomerDB 
# 
# `Unnamed` - Column without a name or meaning <br>
# `Loyalty#` - Loyalty program identifier for the customer <br>
# `First Name` - Customer's first name <br>
# `Last Name` - Customer's last name <br>
# `Customer Name` - Full name of the customer <br>
# `Country` - Country where the customer resides <br>
# `Province or State` - Province or state of the customer's residence <br>
# `City` - City of the customer's residence <br>
# `Latitude` - Latitude coordinate of the customer's location <br>
# `Longitude` - Longitude coordinate of the customer's location <br>
# `Postal code` - Postal code of the customer's address <br>
# `Gender` - Gender of the customer <br>
# `Education` - Education level of the customer <br>
# `Location Code` - Code representing the customer's location <br>
# `Income` - Income level of the customer <br>
# `Marital Status` - Marital status of the customer <br>
# `LoyaltyStatus` - Status of the customer's loyalty program <br>
# `EnrollmentDateOpening` - Date when the customer enrolled in the loyalty program <br>
# `CancellationDate` - Date when the customer canceled their loyalty program <br>
# `Customer Lifetime Value` - Total value of the customer over their lifetime <br>
# `EnrollmentType` - Type of enrollment in the loyalty program <br>
# 
# 
# FlightsDB
# 
# `Loyalty#` - Loyalty program identifier for the customer <br>
# `Year` - Year of the flight activity <br>
# `Month` - Month of the flight activity <br>
# `YearMonthDate` - Date representing the year and month of the flight activity <br>
# `NumFlights` - Number of flights taken by the customer <br>
# `NumFlightsWithCompanions` - Number of flights taken with companions <br>
# `DistanceKM` - Total distance traveled in kilometers <br>
# `PointsAccumulated` - Points accumulated by the customer <br>
# `PointsRedeemed` - Points redeemed by the customer <br>
# `DollarCostPointsRedeemed` - Dollar cost of the points redeemed <br>
# 

# %% [markdown]
# ## 1.1. Importing Libraries <a class="anchor" id="importing-libraries"></a>

# %%
# --- Install required packages 

# pip install pandas 
# pip install numpy 
# pip install matplotlib 
# pip install seaborn 
# pip install plotly 
# pip install pyyaml
# pip install geopandas

# %%
# --- Standard Imports
import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import plotly.express as px
import matplotlib
import math
import matplotlib as mpl
from cycler import cycler

# %%
# Plot style configuration (unified look)
# Keeps logic intact; only global appearance defaults are set.

# Accessible, consistent palette
_palette = sns.color_palette("colorblind", 10)
_palette_hex = _palette.as_hex()

# Seaborn / Matplotlib defaults
sns.set_theme(style="whitegrid", context="notebook", palette=_palette)
mpl.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.titleweight': 'semibold',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'axes.grid': True,
    'grid.color': '#EAEAEA',
    'grid.linestyle': '-',
    'grid.alpha': 0.6,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.borderpad': 0.4,
    'legend.loc': 'best',
    'lines.linewidth': 2.0,
    'axes.prop_cycle': cycler('color', _palette_hex),
})

# Plotly defaults (to align with seaborn/matplotlib)
px.defaults.template = 'plotly_white'
px.defaults.color_discrete_sequence = _palette_hex
px.defaults.color_continuous_scale = 'Cividis'
px.defaults.width = 900
px.defaults.height = 500


# %% [markdown]
# ## 1.2. loading and Reading Data <a class="anchor" id="loading-and-reading-data"></a>

# %%
# directory with raw CSV files
data_dir = Path("../data/raw")

# list all CSV files
csv_files = list(data_dir.glob("*.csv"))
print(f"Found CSV files: {[f.name for f in csv_files]}")


# %%
# specify the files we want to load

customers_file = data_dir / "DM_AIAI_CustomerDB.csv"
flights_file   = data_dir / "DM_AIAI_FlightsDB.csv"

# load them with pandas
customers = pd.read_csv(customers_file)
flights   = pd.read_csv(flights_file)

print("Customers shape:", customers.shape)
print("Flights shape:", flights.shape)

# %% [markdown]
# ## 1.3. Brief Preliminary Analysis <a class="anchor" id="brief-preliminary-analysis"></a>

# %%
customers.columns

# %%
customers['Loyalty#'].count()

# %%
customers.info

# %%
customers.describe().T

# %%
customers.describe(include='object').T

# %%
flights.columns

# %%
flights.info()

# %%
flights.describe().T

# %%
flights.describe(include='object').T

# %% [markdown]
# Remove the column "Unnamed: 0" because it is not needed and has no context in the data.

# %%
if "Unnamed: 0" in customers.columns:
	customers.drop(columns=["Unnamed: 0"], inplace=True)

# %% [markdown]
# # 2. Missing Values and Data Validity Checks  <a class="anchor" id="data-validity-checks"></a>

# %% [markdown]
# Some text to describe what we are doing here. Explanation of this section. We only check smissing values , because in the first deliverable. we are only allowed to notice them and do little cleaning .but we need to notice them .

# %%
# Converting the datatype of categorical features from object to category

flights['YearMonthDate'] = pd.to_datetime(flights['YearMonthDate'], format='%m/%d/%Y')


# Convert NumFlights and NumFlightsWithCompanions to integer type (they are float by default)
flights["NumFlights"] = flights["NumFlights"].astype(int)
flights["NumFlightsWithCompanions"] = flights["NumFlightsWithCompanions"].astype(int)

# %%
# Check the datatype of YearMonthDate column
flights['YearMonthDate'].dtype

# %% [markdown]
# ## 2.1. Missing Values <a class="anchor" id="missing-values"></a>

# %% [markdown]
# Some text to describe what we are doing here.

# %%
# Check for missing values
print("Missing values in CustomerDB:")
display(customers.isna().sum())

print("\nMissing values in FlightsDB:")
display(flights.isna().sum())


# %%
missing_ratio = customers.isna().mean().sort_values(ascending=False)
print((missing_ratio * 100).round(2))

# %% [markdown]
# The flights dataset is clean. The customers dataset has missing values in the columns "CancellationDate", "Customer Lifetime Value" and "Income". We start with analyzing the CancellationDate column. We expect that, these missing values correspond to customers who have not cancelled their subscription. We will check this assumption by looking at some entries with missing CancellationDate values.

# %%
# select first 10 rows where CancellationDate is NaN
active_customers = customers[customers["CancellationDate"].isna()].head(10)
active_customers

# %% [markdown]
# The Entries seem to look fine. The expectation seems to be correct. We will add a new column in the Data Engineering section to indicate whether a customer is active or not based on the CancellationDate column.

# %% [markdown]
# Next we check the missing values in the "Customer Lifetime Value" and "Income" columns. 

# %%
# rows where Income is missing
missing_income = customers[customers["Income"].isna()]
print("Rows with missing Income:", missing_income.shape[0])
display(missing_income.head(5))

# %%
# rows where Customer Lifetime Value is missing
missing_clv = customers[customers["Customer Lifetime Value"].isna()]
print("Rows with missing Customer Lifetime Value:", missing_clv.shape[0])
display(missing_clv.head(5))

# %%
# Check if the missing values in Income and Customer Lifetime Value overlap
missing_income_indices = set(missing_income.index)
missing_clv_indices = set(missing_clv.index)
overlapping_indices = missing_income_indices.intersection(missing_clv_indices)

print("Rows with missing values in both Income and Customer Lifetime Value:", len(overlapping_indices))

# %% [markdown]
# The missing values in Income and Customer Lifetime Value do overlap fully. This means that all rows with missing Income also have missing Customer Lifetime Value. This could indicate a correlation between the two columns, or it could be due to data collection issues. Only 20 rows are affected, which is a small fraction of the total dataset. Deleting these rows might be a reasonable approach for the second part of the project. But for now we will keep them as is.

# %% [markdown]
# ## 2.2. Validity check <a class="anchor" id="validity-check"></a>

# %% [markdown]
# We check all the columns for validity of the values. 

# %% [markdown]
# ### 2.2.1 Categorical Features of customers <a class="anchor" id="categorical-features"></a>

# %%
# List all categorical columns in customers
cat_columns_customers = customers.select_dtypes(include=['object']).columns.tolist()
cat_columns_customers

# %%
# Check the categorical columns for unexpected values
for col in ["Gender","Country", "Education", "Marital Status","LoyaltyStatus", "Province or State", "City", "EnrollmentType"]:
    if col in customers.columns:
        print(f"\n{col} categories:")
        print(customers[col].value_counts(dropna=False))

# %%
# Check for postal codes
print("Unique postal codes:", customers["Postal code"].nunique())
print(customers["Postal code"].value_counts().head(10))

#Check if Postal code has entries with more or less than 7 characters
long_postal_codes = customers[customers["Postal code"].astype(str).str.len() > 7]
print("Postal codes with more than 7 characters:", long_postal_codes.shape[0])
short_postal_codes = customers[customers["Postal code"].astype(str).str.len() < 7]
print("Postal codes with less than 7 characters:", short_postal_codes.shape[0])


# %% [markdown]
# Check of the unique values in categorical columns to see if they make sense.
# 
# - Gender: Only male and female --> valid
# - Country: Only Canada --> valid
# - Education: Only valid education levels --> valid
# - Marital Status: Only valid marital statuses --> valid
# - LoyaltyStatus: Only valid loyalty statuses --> valid
# - Province or State: Only valid provinces or states in Canada --> valid
# - City: Only valid cities in Canada --> valid
# - EnrollmentType: Only valid enrollment types --> valid

# %% [markdown]
# ### 2.2.2 Numerical Features of customers <a class="anchor" id="numerical-features-customers"></a>

# %%
# Min and Max Values of each column of customers db
numeric_cols = customers.select_dtypes(include=["number"]).columns

min_max = customers[numeric_cols].agg(["min", "max"]).T
min_max.transpose()


# %% [markdown]
# - Loyalty#: Is the unique identifier for each customer. Check if unique
# - Lattitude and Longitude make sense because it is only data from Canada (Longitutde: -52 to -141, Latitude: 41 to 83)
# - Income has no negative values, but needs to be checked for outliers
# - Customer Lifetime Value has no negative values, but needs to be checked for outliers
# - IsActive makes sense because it is only 0 and 1
# 
# 
# Next we check for unique loyalty# 

# %%
# Check uniqueness of Loyalty in customers
n_rows = customers.shape[0]
n_unique_ids = customers["Loyalty#"].nunique()

print("Rows in CustomerDB:", n_rows)
print("Unique Loyalty IDs:", n_unique_ids)

if n_rows == n_unique_ids:
    print("Loyalty is unique per row.")
else:
    print ("There are", n_rows - n_unique_ids - 1, "duplicated Loyalty values.")


# %% [markdown]
# - There are some duplicated Loyalty values in the customers dataset. We have to check if these are really duplicates or if there are different customers with the same loyalty id. If they are duplicates, we have to delete them.

# %%
# List of the inconsistent Loyalty IDs in customers dataset
duplicates_by_name = customers[customers.duplicated(subset=["Loyalty#"], keep=False)].sort_values(by=["Loyalty#"])
duplicates_by_name[['Loyalty#', 'First Name', 'Last Name']]

# %% [markdown]
# - That means that different customers have the same loyalty id. This is a problem because we cannot identify the customers uniquely and can not map their flights correctly. We check how many entries in flights correspond to these inconsistent loyalty ids.
#   

# %%
# Count the number of entries in flights corresponding to the inconsistent loyalty ids in total
inconsistent_loyalty_ids = duplicates_by_name['Loyalty#'].unique()
flights_inconsistent = flights[flights['Loyalty#'].isin(inconsistent_loyalty_ids)]
flights_inconsistent_count = flights_inconsistent['Loyalty#'].value_counts()
flights_inconsistent_count.sum()
print("Number of flight entries with inconsistent loyalty IDs:", flights_inconsistent_count.sum())

# Percentage of flights affected by inconsistent loyalty ids
total_flights = flights.shape[0]
affected_percentage = (flights_inconsistent_count.sum() / total_flights) * 100
print(f"Percentage of flights affected by inconsistent loyalty IDs: {affected_percentage:.2f}%")

# %% [markdown]
# - One Major problem of these duplicated loyalty ids is, that we cannot link the customers to their flights. There is no reasonable way to differentiate between the two customers with the same loyalty id. So that the even changing the loyalty id of one customer would not help, because we still do not know which flights belong to which customer. We would create a new customer with a new loyalty id, but this customer would have no flights linked to him. So we would loose all the information about the flights of these customers and the other customer would have all the flights linked to him. This would distort the data and make it unusable for analysis.
#   
# - So for this reason we have to delete all customers with duplicated loyalty ids and all their flights. This is the only way to ensure that the remaining data is valid and can be used for analysis.
# 
# - We may lose some customers, but there is no other way to ensure the validity of the data. The percentage of affected customers is 1.93%, which is relatively low compared to the total number of customers, so the impact on the analysis should be minimal.

# %%
# Clean the customers and flights datasets by removing entries with duplicated Loyalty IDs
loyalty_ids_to_remove = duplicates_by_name['Loyalty#'].unique()
customers = customers[~customers['Loyalty#'].isin(loyalty_ids_to_remove)]
flights = flights[~flights['Loyalty#'].isin(loyalty_ids_to_remove)]
print("Cleaned Customers shape:", customers.shape)
print("Cleaned Flights shape:", flights.shape)

# %% [markdown]
# ### 2.2.3 Numerical Features of flights <a class="anchor" id="numerical-features-flights"></a>

# %%
# Min and Max Values of each column of flights db
numeric_cols = flights.select_dtypes(include=["number"]).columns

min_max = flights[numeric_cols].agg(["min", "max"]).T
min_max.transpose()

# %% [markdown]
# Explanation of the validity check:
# 
# - Year is between 2019 and 2021 --> makes sense
# - Month is between 1 and 12 --> makes sense
# - Day is between 1 and 31 --> makes sense
# - NumFlights is between 0 and 21 --> makes sense 
# - NumFlightsWithCompanions is between 0 and 11 --> makes sense
# - DistanceKM is between 0 and 42040 --> makes sense
# - PointsAccumulated is between 0 and 42040 --> makes sense
# - PointsRedeemed is between 0 and 7496 --> makes sense
# - DollarCostPointsRedeemed is between 0 and 74 --> makes sense

# %% [markdown]
# Validity check of YearMonthDate column:

# %% [markdown]
# ### 2.2.4 Datetime Features of Flights <a class="anchor" id="datetime-features-flights"></a>

# %%
# Print all the unique values of YearMonthDate column sorted by date and their freuquency
flights['YearMonthDate'].value_counts().sort_index()

# %% [markdown]
# Since the Dataset is from 2019 to 2021, the dates should be between 01/01/2019 and 12/31/2021. The unique values of the column confirm this.

# %% [markdown]
# # 2.3 Logic Checks <a class="anchor" id="logic-checks"></a>
# 

# %% [markdown]
# In this Section we perform logic checks on the data to ensure that the values in the columns make sense in relation to each other. We check for inconsistencies and anomalies that could indicate errors in the data.

# %% [markdown]
# ### 2.3.1 Logic Check - DistanceKM and NumFlights <a class="anchor" id="distancekm-and-numflights"></a>
# 

# %% [markdown]
# We want to check if there are any entries where NumFlights is 0 but DistanceKM is greater than 0. This would be a logical inconsistency, as it would imply that the customer traveled a distance without taking any flights.

# %%
# Check flights for DistanceKM and NumFlights. If NumFlights is 0, DistanceKM should also be 0 and vice versa.
# Check first how many entries have NumFlights = 0 but DistanceKM > 0
inconsistent_distance = flights[(flights["NumFlights"] == 0) & (flights["DistanceKM"] > 0)]
print("Entries with NumFlights = 0 but DistanceKM > 0:", inconsistent_distance.shape[0])
display(inconsistent_distance.head(3))

# Check first how many entries have DistanceKM = 0 but NumFlights > 0
inconsistent_numflights = flights[(flights["DistanceKM"] == 0) & (flights["NumFlights"] > 0)]
print("Entries with DistanceKM = 0 but NumFlights > 0:", inconsistent_numflights.shape[0])
display(inconsistent_numflights.head(3))

# %% [markdown]
# ### 2.3.2 Logic Check - NumFlights and FlightCompanions <a class="anchor" id="numflights-vs-companions"></a>
# 

# %% [markdown]
# Customers who have companions on flights should have at least one flight. And the number of flights with companions should not exceed the total number of flights. We want to check if there are any entries where NumFlightsWithCompanions is greater than 0 but NumFlights is 0. Furthermore we check if there are any entries where NumFlightsWithCompanions is greater than NumFlights. If we check for the second Condition, then this would include the first condition This would be a logical inconsistency, as it would imply that the customer has companions on flights without taking any flights themselves. 

# %%
# Check for entries where NumFlightsWithCompanions > NumFlights
inconsistent_companions = flights[(flights["NumFlightsWithCompanions"] > flights["NumFlights"])]
print("Entries with NumFlightsWithCompanions > NumFlights:", inconsistent_companions.shape[0])
display(inconsistent_companions.head(3))


# %% [markdown]
# We have found some inconsistent entries. We notice them here. We will try to fix them with imputation in the second part of the project. For now we just identify them. We will do some additional logic checks after feature engineering in the next section.

# %% [markdown]
# # 3. Feature Engineering  <a class="anchor" id="feature-engineering"></a>

# %% [markdown]
# In this section, we will perform feature engineering to enhance our dataset for better analysis. This includes adding new features based on existing data.

# %% [markdown]
# ## 3.1 Active Customers <a class="anchor" id="active-customers"></a>

# %% [markdown]
# We create a new column "is_active" in the customers dataframe to indicate whether a customer is active or not based on the CancellationDate column. If the CancellationDate is missing (NaT), we assume the customer is still active and set is_active to 1. Otherwise, we set it to 0. We look at the distribution of the newly created column to see how many active and inactive customers we have.

# %%
customers = customers.copy()   # <-- crucial to avoid SettingWithCopyWarning

# New column "IsActive": 1 if CancellationDate is NaN, else 0
customers["IsActive"] = customers["CancellationDate"].isna().astype(int)
# percentage distribution
customers["IsActive"].value_counts(normalize=True).round(3) * 100

# %% [markdown]
# ## 3.2 Tenure Days  <a class="anchor" id="tenure-days"></a>

# %% [markdown]
# Tenure Days is calculated by the difference between EnrollmentDate and CancellationDate. If the CancellationDate is missing, we assume the customer is still active and use the last date in the dataset (31/12/2021) for the calculation.

# %%
customers = customers.copy()   # <-- crucial to avoid SettingWithCopyWarning

today = pd.Timestamp('2021-12-31').normalize()

# Safe conversions
customers['EnrollmentDateOpening'] = pd.to_datetime(customers['EnrollmentDateOpening'], errors='coerce')
customers['CancellationDate']      = pd.to_datetime(customers['CancellationDate'], errors='coerce')

# EndDate: CancellationDate or today
customers['EndDate'] = customers['CancellationDate'].fillna(today)

# Tenure in days
customers['CustomerTenureDays'] = (customers['EndDate'] - customers['EnrollmentDateOpening']).dt.days

# Drop temp column (avoid inplace on possible views)
customers = customers.drop(columns=['EndDate'])

# Head of the updated customers dataframe with the columns related to tenure
customers[['EnrollmentDateOpening', 'CancellationDate', 'CustomerTenureDays']].head()


# %% [markdown]
# ## 3.3 Flight Metrics per Customer  <a class="anchor" id="flight-metrics-per-customer"></a>
# 

# %% [markdown]
# - We will add some flight metrics grouped by customer into the customers dataframe. First we calculate them in a new dataframe and then we merge them into the customers dataframe.

# %% [markdown]
# ### 3.3.1 Flights Aggregated Metrics  <a class="anchor" id="flights-aggregated-metrics"></a>
# 
# 

# %%
# Basic flight metrics per customer
flight_aggs = flights.groupby('Loyalty#').agg(
    # Aggregated Values
    total_flights=('NumFlights', 'sum'),
    total_flights_with_companions=('NumFlightsWithCompanions', 'sum'), 
    total_distance=('DistanceKM', 'sum'),
    total_points_accumulated=('PointsAccumulated', 'sum'),
    total_points_redeemed=('PointsRedeemed', 'sum'),
    total_cost_redeemed=('DollarCostPointsRedeemed', 'sum'),
    average_distance_per_flight=('DistanceKM', lambda x: round(x.sum() / x.count(), 2) if x.count() > 0 else 0)
).reset_index()

print("Flight aggregations summary:")

# Add derived metrics
flight_aggs['points_redemption_ratio'] = (
    flight_aggs['total_points_redeemed'] / 
    flight_aggs['total_points_accumulated'].replace(0, np.nan)
).fillna(0)

flight_aggs['companion_flight_ratio'] = (
    flight_aggs['total_flights_with_companions'] / 
    flight_aggs['total_flights']
).fillna(0)

display(flight_aggs.head())


# %% [markdown]
# ### 3.3.2 Merge Flights Aggregated with Customers  <a class="anchor" id="merge-flights-aggregated-with-customers"></a>
# 
# 

# %%
# Add the aggregated data from flights to the customer data
customers = customers.merge(flight_aggs, on='Loyalty#', how='left')

# Show the first 5 rows of the updated customers dataframe
display(customers.head())

# %% [markdown]
# ## 3.4 Season to flights dataset <a class="anchor" id="flight-metrics-per-customer"></a>
# 

# %% [markdown]
# In this Section we add the Column "Season" to the flights dataset based on the Month column. We define the seasons as follows:
# - Winter: December, January, February
# - Spring: March, April, May
# - Summer: June, July, August
# - Autumn: September, October, November
# 
# 
# We will use this in our Behavioral Segmentation analysis later.

# %%
# Add Column "Season" to flights dataset based on the Month column
def month_to_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return np.nan
    
flights['Month'] = flights['YearMonthDate'].dt.month
flights['Season'] = flights['Month'].apply(month_to_season) 
flights[['YearMonthDate', 'Month', 'Season']].head()

# Show the first 5 rows of the updated flights dataframe
display(flights.head())

# %% [markdown]
# ## 3.5 Logic Checks of the new columns <a class="anchor" id="logic-checks-new-columns"></a>
# 

# %% [markdown]
# After Feature Engineering, we perform logic checks on the newly created columns to ensure their validity. We already did that in Section 2.3, but we will repeat the checks here to confirm that the new features are consistent with the existing data.

# %% [markdown]
# ### 3.5.1 Logic Check - Customer Lifetime Value and Zero Flights <a class="anchor" id="clv-zero-flights"></a>
# 
# 

# %% [markdown]
# We should check if there are any entries where Customer Lifetime Value is greater than 0 but NumFlights is 0. This would be a logical inconsistency, as it would imply that the customer has a lifetime value without taking any flights. So far we dont have a lot of information, how the Customer Lifetime Value is calculated. But it is reasonable to assume, that a customer who has not taken any flights should not have any lifetime value.

# %%
# Check for entries where Customer Lifetime Value > 0 but NumFlights = 0
inconsistent_clv = customers[(customers["Customer Lifetime Value"] > 0) & (customers["total_flights"] == 0)]
print("Entries with Customer Lifetime Value > 0 but NumFlights = 0:", inconsistent_clv.shape[0])
display(inconsistent_clv.head(3))

# %% [markdown]
# ### 3.5.2 Logic Check - AverageDistanceKM > 13,000 KM <a class="anchor" id="avg-distancekm-over-13000km"></a>

# %% [markdown]
# We want to check if there are any entries where a single flight is greater than 13,000 KM. DistanceKM is accumulated, but the average distance per flight should not exceed this value. The maximum distance between any two points in Canada is approximately 13,000 KM. If there are entries with AverageDistanceKM greater than this value, it would indicate a potential data error.

# %%
# Maximum possible distance for a single flight (approx. Vancouver → Singapore)
max_km_per_flight = 13000  

# Check for entries where average_distance_per_flight > 13,000 KM
inconsistent_distance = customers[customers["average_distance_per_flight"] > max_km_per_flight]
print("Entries with average_distance_per_flight > 13,000 KM:", inconsistent_distance.shape[0])
display(inconsistent_distance.head(3))

# %% [markdown]
# ### 3.5.3 Logic Check - Negative Customer Tenure Days <a class="anchor" id="negative-tenure-days"></a>

# %% [markdown]
# We want to check if there are entries, where the Customer Tenure Days is negative. This would indicate a data error, as the CancellationDate should always be after the EnrollmentDate.

# %%
# Check for entries where the Customer Tenure Days is negative
inconsistent_tenure = customers[customers["CustomerTenureDays"] < 0]
print("Entries with negative Customer Tenure Days:", inconsistent_tenure.shape[0])
display(inconsistent_tenure.head(3))

# %% [markdown]
# # 4. Exploratory Data Analysis (EDA)  <a class="anchor" id="eda"></a>

# %% [markdown]
# We now will analyze the two dataset. We will split the notebook into 4.1 for customers and 4.2 for flights.

# %% [markdown]
# ## 4.1 Customer EDA  <a class="anchor" id="Customers-EDA"></a>
# 

# %% [markdown]
# We split the Customer EDA into categorical and numerical features for better analysis.
# 
# - Integer and Float - Histogram, Box Plot, Histogram related with the target variable.
# - Categories - Histogram

# %% [markdown]
# ### 4.1.1 Customer EDA - Categorical Features  <a class="anchor" id="Customers-EDA-Categorical"></a>
# 

# %%
# Filter for only categorical features
cat_columns = customers.select_dtypes(include=['object', 'category']).columns.tolist()
customers_cat = customers[cat_columns]

# Remove First Name, Last Name and Customer Name from the list --> they are not useful for EDA
customers_cat = customers_cat.drop(columns=['First Name', 'Last Name', 'Customer Name'], errors='ignore')


# %%
# Grid Layout for Categorical Features
n_cols = 2
n_rows = math.ceil(len(customers_cat.columns) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]

for i, col in enumerate(customers_cat.columns):
    ax = axes[i]
    value_counts = customers_cat[col].value_counts(dropna=False)

    # Limit number of categories for readability
    max_categories = 10
    if len(value_counts) > max_categories:
        top_cats = value_counts.head(max_categories)
        data = customers_cat[customers_cat[col].isin(top_cats.index)]
        order = top_cats.index
        ax.set_title(f'Distribution of {col} (Top {max_categories})')
    else:
        data = customers_cat
        order = value_counts.index
        ax.set_title(f'Distribution of {col}')

    sns.countplot(data=data, x=col, order=order, ax=ax)

    # Rotate and shrink if necessary
    label_count = len(order)
    label_size = 10 if label_count <= 6 else 8 if label_count <= 10 else 6
    ax.tick_params(axis='x', labelrotation=45, labelsize=label_size)

    # Right-align for readability
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')

# Hide empty axes
for j in range(len(customers_cat.columns), len(axes)):
    fig.delaxes(axes[j])

sns.despine(fig)
plt.tight_layout()
plt.show()


# %% [markdown]
# ### 4.1.2 Customer EDA - Numerical Features  <a class="anchor" id="Customers-EDA-Numerical"></a>
# 

# %%
# Filter for only numerical features
num_columns = customers.select_dtypes(include=np.number).columns.tolist()
customers_num = customers[num_columns]

# Remove Loyalty#, Latitude, Longitude from the list --> they are not useful for EDA
customers_num = customers_num.drop(columns=['Loyalty#', 'Latitude', 'Longitude'], errors='ignore')


# %%
# Create a figure with 3 columns (hist, box, kde)
n_features = len(customers_num.columns)
n_cols = 3
n_rows = n_features

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))

# If there’s only one feature, make axes 2D
if n_rows == 1:
    axes = np.expand_dims(axes, axis=0)

for i, col in enumerate(customers_num.columns):
    # Histogram
    sns.histplot(customers_num[col], bins=30, kde=False, ax=axes[i, 0], color='skyblue')
    axes[i, 0].set_title(f'Histogram of {col}')
    axes[i, 0].set_xlabel(col)
    axes[i, 0].set_ylabel('Count')

    # Boxplot
    sns.boxplot(x=customers_num[col], ax=axes[i, 1], color='lightcoral')
    axes[i, 1].set_title(f'Boxplot of {col}')
    axes[i, 1].set_xlabel(col)
    axes[i, 1].set_ylabel('')

    # Density plot (KDE)
    sns.kdeplot(customers_num[col].dropna(), ax=axes[i, 2], fill=True, color='mediumseagreen')
    axes[i, 2].set_title(f'Density Plot of {col}')
    axes[i, 2].set_xlabel(col)
    axes[i, 2].set_ylabel('Density')

plt.tight_layout()
sns.despine(fig)
plt.show()


# %% [markdown]
# ### 4.1.3 Customer EDA - Multivariate Analysis  <a class="anchor" id="customers-eda-multi-variate"></a>
# 

# %% [markdown]
# We do a multi-variate analysis of the numerical features to see if there are any correlations between them.

# %% [markdown]
# We start with a correlation heatmap to see the correlations between the numerical features

# %%
corr = customers_num.corr(method="pearson")

# mask upper triangle for readability
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(min(1.2*len(customers_num), 22), min(1.0*len(customers_num), 18)))
sns.heatmap(
    corr,
    mask=mask,
    cmap="coolwarm",
    center=0,
    annot=True,         
    fmt=".2f",           # show 2 decimal places
    square=True,
    cbar_kws={"shrink": 0.75},
)
plt.title("Correlation Heatmap (Pearson) — Numeric Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()


# %% [markdown]
# We continue with pairplots to see the relationships between the numerical and categorical features.
# We do it for every categorical feature with a selected numerical feature (plotting every numerical feature would be too much).
# 
# 
# --> Write here more explanations of the coming plots

# %%
# Gender vs Numeric Features: Violinplots (with internal box)
cat_for_group = "Gender"

# List of numeric columns 
num_cols = ('Income', 'Customer Lifetime Value', 'CustomerTenureDays', 'total_flights', 'total_distance')

if cat_for_group in customers.columns and len(num_cols) > 0:
    # Order categories by frequency for consistent x-axis across rows
    cat_order = customers[cat_for_group].value_counts().index.tolist()

    n_rows = len(num_cols)
    fig_height = max(3.2 * n_rows, 4)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, fig_height), squeeze=False)
    sns.set(style="whitegrid")

    for i, col in enumerate(num_cols):
        ax = axes[i, 0]
        sns.violinplot(
            data=customers,
            x=cat_for_group,
            y=col,
            order=cat_order,
            inner="box",   
            cut=0,
            ax=ax
        )
        ax.set_title(f"{col} by {cat_for_group}", fontsize=12)
        ax.set_xlabel(cat_for_group)
        ax.set_ylabel(col)
        ax.tick_params(axis='x', labelrotation=25)

    fig.suptitle(f"Numeric vs {cat_for_group}: Violinplots with Box", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.show()
else:
    print(f"'{cat_for_group}' not found in customers or no numeric columns detected.")


# %%
# Education vs Numeric Features: Violinplots (with internal box)
cat_for_group = "Education"

# List of numeric columns 
num_cols = ('Income', 'Customer Lifetime Value', 'CustomerTenureDays', 'total_flights', 'total_distance')

if cat_for_group in customers.columns and len(num_cols) > 0:
    # Order categories by frequency for consistent x-axis across rows
    cat_order = customers[cat_for_group].value_counts().index.tolist()

    n_rows = len(num_cols)
    fig_height = max(3.2 * n_rows, 4)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, fig_height), squeeze=False)
    sns.set(style="whitegrid")

    for i, col in enumerate(num_cols):
        ax = axes[i, 0]
        sns.violinplot(
            data=customers,
            x=cat_for_group,
            y=col,
            order=cat_order,
            inner="box",   
            cut=0,
            ax=ax
        )
        ax.set_title(f"{col} by {cat_for_group}", fontsize=12)
        ax.set_xlabel(cat_for_group)
        ax.set_ylabel(col)
        ax.tick_params(axis='x', labelrotation=25)

    fig.suptitle(f"Numeric vs {cat_for_group}: Violinplots with Box", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.show()
else:
    print(f"'{cat_for_group}' not found in customers or no numeric columns detected.")


# %%
# Education vs Numeric Features: Violinplots (with internal box)
cat_for_group = "Location Code"

# List of numeric columns 
num_cols = ('Income', 'Customer Lifetime Value', 'CustomerTenureDays', 'total_flights', 'total_distance')

if cat_for_group in customers.columns and len(num_cols) > 0:
    # Order categories by frequency for consistent x-axis across rows
    cat_order = customers[cat_for_group].value_counts().index.tolist()

    n_rows = len(num_cols)
    fig_height = max(3.2 * n_rows, 4)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, fig_height), squeeze=False)
    sns.set(style="whitegrid")

    for i, col in enumerate(num_cols):
        ax = axes[i, 0]
        sns.violinplot(
            data=customers,
            x=cat_for_group,
            y=col,
            order=cat_order,
            inner="box",   
            cut=0,
            ax=ax
        )
        ax.set_title(f"{col} by {cat_for_group}", fontsize=12)
        ax.set_xlabel(cat_for_group)
        ax.set_ylabel(col)
        ax.tick_params(axis='x', labelrotation=25)

    fig.suptitle(f"Numeric vs {cat_for_group}: Violinplots with Box", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.show()
else:
    print(f"'{cat_for_group}' not found in customers or no numeric columns detected.")


# %%
# Education vs Numeric Features: Violinplots (with internal box)
cat_for_group = "Marital Status"

# List of numeric columns 
num_cols = ('Income', 'Customer Lifetime Value', 'CustomerTenureDays', 'total_flights', 'total_distance')

if cat_for_group in customers.columns and len(num_cols) > 0:
    # Order categories by frequency for consistent x-axis across rows
    cat_order = customers[cat_for_group].value_counts().index.tolist()

    n_rows = len(num_cols)
    fig_height = max(3.2 * n_rows, 4)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, fig_height), squeeze=False)
    sns.set(style="whitegrid")

    for i, col in enumerate(num_cols):
        ax = axes[i, 0]
        sns.violinplot(
            data=customers,
            x=cat_for_group,
            y=col,
            order=cat_order,
            inner="box",   
            cut=0,
            ax=ax
        )
        ax.set_title(f"{col} by {cat_for_group}", fontsize=12)
        ax.set_xlabel(cat_for_group)
        ax.set_ylabel(col)
        ax.tick_params(axis='x', labelrotation=25)

    fig.suptitle(f"Numeric vs {cat_for_group}: Violinplots with Box", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.show()
else:
    print(f"'{cat_for_group}' not found in customers or no numeric columns detected.")


# %%
# Education vs Numeric Features: Violinplots (with internal box)
cat_for_group = "LoyaltyStatus"

# List of numeric columns 
num_cols = ('Income', 'Customer Lifetime Value', 'CustomerTenureDays', 'total_flights', 'total_distance')

if cat_for_group in customers.columns and len(num_cols) > 0:
    # Order categories by frequency for consistent x-axis across rows
    cat_order = customers[cat_for_group].value_counts().index.tolist()

    n_rows = len(num_cols)
    fig_height = max(3.2 * n_rows, 4)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, fig_height), squeeze=False)
    sns.set(style="whitegrid")

    for i, col in enumerate(num_cols):
        ax = axes[i, 0]
        sns.violinplot(
            data=customers,
            x=cat_for_group,
            y=col,
            order=cat_order,
            inner="box",   
            cut=0,
            ax=ax
        )
        ax.set_title(f"{col} by {cat_for_group}", fontsize=12)
        ax.set_xlabel(cat_for_group)
        ax.set_ylabel(col)
        ax.tick_params(axis='x', labelrotation=25)

    fig.suptitle(f"Numeric vs {cat_for_group}: Violinplots with Box", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.show()
else:
    print(f"'{cat_for_group}' not found in customers or no numeric columns detected.")


# %%
# Education vs Numeric Features: Violinplots (with internal box)
cat_for_group = "EnrollmentType"

# List of numeric columns 
num_cols = ('Income', 'Customer Lifetime Value', 'CustomerTenureDays', 'total_flights', 'total_distance')

if cat_for_group in customers.columns and len(num_cols) > 0:
    # Order categories by frequency for consistent x-axis across rows
    cat_order = customers[cat_for_group].value_counts().index.tolist()

    n_rows = len(num_cols)
    fig_height = max(3.2 * n_rows, 4)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, fig_height), squeeze=False)
    sns.set(style="whitegrid")

    for i, col in enumerate(num_cols):
        ax = axes[i, 0]
        sns.violinplot(
            data=customers,
            x=cat_for_group,
            y=col,
            order=cat_order,
            inner="box",   
            cut=0,
            ax=ax
        )
        ax.set_title(f"{col} by {cat_for_group}", fontsize=12)
        ax.set_xlabel(cat_for_group)
        ax.set_ylabel(col)
        ax.tick_params(axis='x', labelrotation=25)

    fig.suptitle(f"Numeric vs {cat_for_group}: Violinplots with Box", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.show()
else:
    print(f"'{cat_for_group}' not found in customers or no numeric columns detected.")


# %% [markdown]
# We can see that the Income Distribution is different: The highest income levels are achieved by Bachelor Degree holders, while Master, Doctor or High school degrees holders tend to be in lower but similar income levels. The College Degree holders all have an income of 0. 

# %% [markdown]
# # 4.2 Flights EDA  <a class="anchor" id="flights-eda"></a>
# 
# 
# 

# %% [markdown]
# We only have numerical features and one pandas datetime feature in the flights dataset. First we filter for the numerical features and do the EDA on them. 

# %% [markdown]
# ### 4.2.1 Flights EDA - Numerical Features  <a class="anchor" id="flights-eda-numerical"></a>

# %%
# Filter for numerical features
num_columns_flights = flights.select_dtypes(include=np.number).columns.tolist()
flights_num = flights[num_columns_flights]

# Remove Loyalty#, Month and Year --> they are not useful for EDA, Month and Year have the same value for all rows
flights_num = flights_num.drop(columns=['Loyalty#', 'Month', 'Year'], errors='ignore')

# %%
# Create a figure with 3 columns (hist, box, kde)
n_features = len(flights_num.columns)
n_cols = 3
n_rows = n_features

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))

# If there’s only one feature, make axes 2D
if n_rows == 1:
    axes = np.expand_dims(axes, axis=0)

for i, col in enumerate(flights_num.columns):
    # Histogram
    sns.histplot(flights_num[col], bins=30, kde=False, ax=axes[i, 0], color='skyblue')
    axes[i, 0].set_title(f'Histogram of {col}')
    axes[i, 0].set_xlabel(col)
    axes[i, 0].set_ylabel('Count')

    # Boxplot
    sns.boxplot(x=flights_num[col], ax=axes[i, 1], color='lightcoral')
    axes[i, 1].set_title(f'Boxplot of {col}')
    axes[i, 1].set_xlabel(col)
    axes[i, 1].set_ylabel('')

    # Density plot (KDE)
    sns.kdeplot(flights_num[col].dropna(), ax=axes[i, 2], fill=True, color='mediumseagreen')
    axes[i, 2].set_title(f'Density Plot of {col}')
    axes[i, 2].set_xlabel(col)
    axes[i, 2].set_ylabel('Density')

plt.tight_layout()
sns.despine(fig)
plt.show()


# %% [markdown]
# # 4.3 Zero income customers EDA  <a class="anchor" id="flights-eda"></a>
# 

# %% [markdown]
# Since we have a lot of customers, who have not taken any flights, we will do a separate EDA for these customers to see if there are any differences between them and the customers who have taken flights.

# %%
zero_income = customers[customers["Income"] == 0]

sns.set(style="whitegrid")


cols = ["Gender", "Marital Status", "Education",
        "Location Code", "EnrollmentType", "LoyaltyStatus"]

fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
axes = axes.ravel()

for ax, c in zip(axes, cols):
    order = zero_income[c].value_counts().index
    sns.countplot(data=zero_income, x=c, order=order, ax=ax)
    ax.set_title(c)
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=30)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{int(h)}",
                        (p.get_x() + p.get_width()/2, h),
                        ha="center", va="bottom", fontsize=8,
                        xytext=(0, 2), textcoords="offset points")

# Common y-label on the leftmost plots; or use a figure-level label:
fig.suptitle("Number of Customers with income = 0 by Category", fontsize=16, fontweight="bold", y=1.02)
fig.text(0.04, 0.5, "Number of Customers", va="center", rotation="vertical")
plt.tight_layout()
plt.show()


# %%
# Calculate counts per city and sort descending
city_counts = zero_income["City"].value_counts().reset_index()
city_counts.columns = ["City", "Count"]

# Keep only Top 10
city_counts = city_counts.head(10)

# Create figure
plt.figure(figsize=(10, 5))

# Bar chart: Number of customers by City (Top 10)
sns.barplot(data=city_counts, x="City", y="Count", palette="viridis")

plt.title("Top 10 Cities with Income = 0")
plt.xlabel("City")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %% [markdown]
# # 5. Deep Down Analysis for the Segmentation  <a class="anchor" id="segmentation-analyses"></a>
# 
# 

# %% [markdown]
# In this section, we will do deeper into our analysis. We don't do the customer segmentation here yet, but we will try to find insights for each segmentation type. We will do Value-Based Segmentation, Behavioral Segmentation and Demographic Segmentation.

# %% [markdown]
# ## 5.1 Value based segmentation   <a class="anchor" id="Flights-EDA"></a>
# 

# %% [markdown]
# In this section we look into Value-Based Segmentation. We will analyze the data based on the income and Customer Lifetime Value of the customers.

# %% [markdown]
# ## 5.1.1 Grouping by Income  <a class="anchor" id="Flights-EDA"></a>
# 

# %% [markdown]
# We start with the Income based segmentation. We create income ranges and group the customers based on these ranges. Then we analyze the income groups. We used the median Income of Canada to create the income ranges. 50% of the population has an income below 40.000 CAD and 50% above. Since we don't know the currency inside the dataset, we assume it is in CAD as well.

# %%
# Create a new column and groupBy Income ranges

ranked = customers['Income'].rank(method='first')  # bricht Ties deterministisch
customers['Income_Group'] = pd.qcut(
    ranked, q=5, labels=['Very Low','Low','Medium','High','Very High']
)

# Get the median Income from Statistca aroung 40k median income
# https://www.statista.com/statistics/464087/median-annual-earnings-in-canada/?srsltid=AfmBOorobKVMkA5XzryvSxoVhMDd2nOgjsvVJ3QwOdzoxHITF-YTxYi1
median_income = 40000

# Define custom bins around the median
bins = [0, 25000, 35000, 45000, 60000, customers['Income'].max()]

labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

# Categorize based on custom bins
customers['Income_Group'] = pd.cut(customers['Income'], bins=bins, labels=labels, include_lowest=True)

# Check the distribution
print(customers['Income_Group'].value_counts().sort_index())

# Optional: check mean/median per group
#print(customers.groupby('Income_Group')['Income'].mean())
#print(customers.groupby('Income_Group')['Income'].median())

# Plot the distribution of Income_Group
plt.figure(figsize=(10, 6))
sns.countplot(data=customers, x='Income_Group', order=labels)
plt.title('Distribution of Income Groups')
plt.xlabel('Income Group')
plt.ylabel('Number of Customers')
plt.show()


# %%
customers['Loyalty#'].count()

# %% [markdown]
# We can see that we have a lot of very low income customers, followed by high income customers. The middle income range has the least customers. This is an interesting finding, as we would expect a more normal distribution of income ranges. This could indicate that the airline is more popular among low and high income customers, while middle income customers are less likely to use the airline. Further analysis is needed to understand the reasons behind this distribution.

# %% [markdown]
# It is important to note, that our income group is already uneven distributed, so we expect similar distributions in the other features as well. We could normalize it, but we prefer not to, so we can see the real distribution of the data.

# %%
# print all columns of customers dataframe
customers.columns.tolist

# %%
# Plot the following: Bar chart. Y-Axis counts the amount of customers. X-Axis is for the categories of total_flights. 
# The total flights categories are 0, 1-10, 10-20, 20-40, 41-100, 101-200, 200+.
# For every income group, plot a separate bar chart with different colors.
plt.figure(figsize=(12, 8))
sns.countplot(
    data=customers,
    x=pd.cut(
        customers['total_flights'],
        bins=[0, 1, 11, 21, 41, 101, 201, float('inf')],
        labels=['0', '1-10', '11-20', '21-40', '41-100', '101-200', '200+'],
        right=False
    ),
    hue='Income_Group',
    order=['0', '1-10', '11-20', '21-40', '41-100', '101-200', '200+']
)
plt.title('Customer Counts by Total Flights and Income Group')
plt.xlabel('Total Flights Category')
plt.ylabel('Number of Customers')
plt.legend(title='Income Group')
plt.show()


# %%
# Plot the following: Bar chart. Y-Axis counts the amount of customers. X-Axis is for the categories of average_distance.
# The average distance categories are 0, 1 - 1000, 1001 - 2000, 2001- 3000, 3001 - 5000, 5001 - 7000, 7001 - 10000, 10001+.
# For every income group, plot a separate bar chart with different colors.
plt.figure(figsize=(12, 8))
sns.countplot(
    data=customers,
    x=pd.cut(
        customers['average_distance_per_flight'],
        bins=[-1, 0, 1000, 2000, 3000, 5000, 7000, 10000, np.inf],
        labels=['0', '1-1000', '1001-2000', '2001-3000', '3001-5000', '5001-7000', '7001-10000', '10001+'],
                        right=True
    ),
    hue='Income_Group',
    order=['0', '1-1000', '1001-2000', '2001-3000', '3001-5000', '5001-7000', '7001-10000', '10001+']
)
plt.title('Customer Counts by Average Distance Per Flight and Income Group')
plt.xlabel('Average Distance Per Flight Category')
plt.ylabel('Number of Customers')
plt.legend(title='Income Group')
plt.show()


# %%
# Boxplot for every income group for Customer Lifetime Value
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=customers,
    x='Income_Group',
    y='Customer Lifetime Value',
    order=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
    palette='Set3'
)
plt.title('Customer Lifetime Value Distribution by Income Group')
plt.xlabel('Income Group')
plt.ylabel('Customer Lifetime Value')
plt.show()

# %%
# Gender distribution per income group 
plt.figure(figsize=(10, 6))
sns.countplot(
    data=customers,
    x='Income_Group',
    hue='Gender',
    order=['Very Low', 'Low', 'Medium', 'High', 'Very High']
)
plt.title('Gender Distribution by Income Group')
plt.xlabel('Income Group')
plt.ylabel('Number of Customers')
plt.legend(title='Gender')
plt.show()

# %%
# Location Code distribution per income group
plt.figure(figsize=(12, 6))
sns.countplot(
    data=customers,
    x='Income_Group',
    hue='Location Code',
    order=['Very Low', 'Low', 'Medium', 'High', 'Very High']
)
plt.title('Location Code Distribution by Income Group')
plt.xlabel('Income Group')
plt.ylabel('Number of Customers')
plt.legend(title='Location Code', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()  

# %%
# Education distribution per income group
plt.figure(figsize=(12, 6))
sns.countplot(
    data=customers,
    x='Income_Group',
    hue='Education',
    order=['Very Low', 'Low', 'Medium', 'High', 'Very High']
)
plt.title('Education Distribution by Income Group')
plt.xlabel('Income Group')
plt.ylabel('Number of Customers')
plt.legend(title='Education')
plt.show()

# %%
# Professional Business-Style Plot: Education Distribution by Income Group
fig, ax = plt.subplots(figsize=(14, 7))

# Business color palette
business_colors = ['#5fbcf4', '#3da8e8', '#0979bd', '#19467e', '#17344e']

# Create grouped bar chart
income_order = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
education_data = customers.groupby(['Income_Group', 'Education']).size().unstack(fill_value=0)
education_data = education_data.reindex(income_order)

# Plot bars
education_data.plot(
    kind='bar',
    ax=ax,
    color=business_colors,
    edgecolor='white',
    linewidth=0.8,
    width=0.75
)

# Styling
ax.set_title('Education Distribution by Income Group', 
             fontsize=16, fontweight='bold', color='#17344e', pad=20)
ax.set_xlabel('Income Group', fontsize=12, fontweight='semibold', color='#17344e')
ax.set_ylabel('Number of Customers', fontsize=12, fontweight='semibold', color='#17344e')

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CCCCCC')
ax.spines['bottom'].set_color('#CCCCCC')

# Grid styling
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5, color='#CCCCCC')
ax.set_axisbelow(True)

# Tick styling
ax.tick_params(colors='#17344e', labelsize=11)
ax.set_xticklabels(income_order, rotation=0, ha='center')

# Legend styling
legend = ax.legend(
    title='Education Level',
    title_fontsize=11,
    fontsize=10,
    loc='upper right',
    frameon=True,
    framealpha=0.95,
    edgecolor='#CCCCCC'
)
legend.get_title().set_color('#17344e')
legend.get_title().set_fontweight('semibold')

# Add value labels on bars (optional - can be removed if too cluttered)
for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', 
                 fontsize=12, padding=3, color='#17344e')

plt.tight_layout()
plt.show()

# %%
# Show some entries of customers with Education Doctor and filter for Income and Customer Lifetime Value

doctorate_customers = customers[customers['Education'] == 'Doctor']
doctorate_customers = doctorate_customers[['Income', 'Customer Lifetime Value', 'Income_Group']]
display(doctorate_customers.head())

# %% [markdown]
# This is really odd: We have a lot of educated people in very low and low income groups. We would expect that higher education levels correlate with higher income levels. Everything above low is only with a Bachelor degree.

# %%
# Marital Status distribution per income group
plt.figure(figsize=(12, 6))
sns.countplot(
    data=customers,
    x='Income_Group',
    hue='Marital Status',
    order=['Very Low', 'Low', 'Medium', 'High', 'Very High']
)
plt.title('Marital Status Distribution by Income Group')
plt.xlabel('Income Group')
plt.ylabel('Number of Customers')
plt.legend(title='Marital Status')
plt.show()

# %%
# Loyalty Status distribution per Income Group as a stacked bar chart 

plt.figure(figsize=(12, 8))
clv_loyalty_counts = customers.groupby(['Income_Group', 'LoyaltyStatus']).size().unstack(fill_value=0)
clv_loyalty_counts.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Loyalty Status Distribution by Income Group')
plt.xlabel('Income Group')
plt.ylabel('Number of Customers')
plt.legend(title='Loyalty Status')
plt.show()

# %%
# Customer Tenure Days distribution per income group
# Filter out negative tenure days if any only for this plot
customers_tenure_days = customers[customers['CustomerTenureDays'] >= 0]
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=customers_tenure_days,
    x='Income_Group',
    y='CustomerTenureDays',
    order=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
    palette='Set2'
)
plt.title('Customer Tenure Days Distribution by Income Group')
plt.xlabel('Income Group')
plt.ylabel('Customer Tenure Days')
plt.show()

# %%
# Enrollment Type distribution per income group
plt.figure(figsize=(12, 6))
sns.countplot(  
    data=customers_tenure_days,
    x='EnrollmentType',
    hue='Income_Group',
    palette='Set2'
)
plt.title('Enrollment Type Distribution by Income Group')
plt.xlabel('Enrollment Type')
plt.ylabel('Count')
plt.legend(title='Income Group')
plt.show()

# %% [markdown]
# ## 5.1.1 Grouping by Customer Lifetime Value  <a class="anchor" id="Customer Lifetime Value"></a>
# 

# %% [markdown]
# Now we group by Customer Lifetime Value ranges and analyze the data based on these groups. We create the ranges based on the quartiles of the Customer Lifetime Value distribution. This way we can see how the different segments of customers behave in relation to their lifetime value.

# %%
# Create a new column and groupBy CLV ranges
ranked_clv = customers['Customer Lifetime Value'].rank(method='first')  # bricht Ties deterministisch
customers['CLV_Group'] = pd.qcut(
    ranked_clv, q=5, labels=['Very Low','Low','Medium','High','Very High']
)

clv_distribution = customers['CLV_Group'].value_counts().sort_index()

# Print the quantile value of CLV_Group
clv_quantiles = customers['Customer Lifetime Value'].quantile([0.2, 0.4, 0.6, 0.8])
print("Customer Lifetime Value Quantiles:")
print(clv_quantiles)

# Plot the distribution of CLV_Group
plt.figure(figsize=(10, 6))
sns.countplot(data=customers, x='CLV_Group', order=['Very Low','Low','Medium','High','Very High'])
plt.title('Distribution of Customer Lifetime Value Groups')
plt.xlabel('Customer Lifetime Value Group')
plt.ylabel('Number of Customers')
plt.show()


# %% [markdown]
# Now that the CLV_Group is evenly distributed, we expect to see more balanced distributions in the other features as well. This will help us to better understand the behavior of customers in relation to their lifetime value without being biased by uneven group sizes.

# %%
# Plot the following: Bar chart. Y-Axis counts the amount of customers. X-Axis is for the categories of total_flights. 
# The total flights categories are 0, 1-10, 10-20, 20-40, 41-100, 101-200, 200+.
# For every CLV_group, plot a separate bar chart with different colors.
plt.figure(figsize=(12, 8))
sns.countplot(
    data=customers,
    x=pd.cut(
        customers['total_flights'],
        bins=[0, 1, 11, 21, 41, 101, 201, float('inf')],
        labels=['0', '1-10', '11-20', '21-40', '41-100', '101-200', '200+'],
        right=False
    ),
    hue='CLV_Group',
    order=['0', '1-10', '11-20', '21-40', '41-100', '101-200', '200+']
)
plt.title('Customer Counts by Total Flights and CLV_Group')
plt.xlabel('Total Flights Category')
plt.ylabel('Number of Customers')
plt.legend(title='CLV_Group')
plt.show()


# %%
# Plot the following: Bar chart. Y-Axis counts the amount of customers. X-Axis is for the categories of average_distance.
# The average distance categories are 0, 1 - 1000, 1001 - 2000, 2001- 3000, 3001 - 5000, 5001 - 7000, 7001 - 10000, 10001+.
# For every CLV_Group, plot a separate bar chart with different colors.
plt.figure(figsize=(12, 8))
sns.countplot(
    data=customers,
    x=pd.cut(
        customers['average_distance_per_flight'],
        bins=[-1, 0, 1000, 2000, 3000, 5000, 7000, 10000, np.inf],
        labels=['0', '1-1000', '1001-2000', '2001-3000', '3001-5000', '5001-7000', '7001-10000', '10001+'],
                        right=True
    ),
    hue='CLV_Group',
    order=['0', '1-1000', '1001-2000', '2001-3000', '3001-5000', '5001-7000', '7001-10000', '10001+']
)
plt.title('Customer Counts by Average Distance Per Flight and CLV_Group')
plt.xlabel('Average Distance Per Flight Category')
plt.ylabel('Number of Customers')
plt.legend(title='CLV_Group')
plt.show()


# %%
# Boxplot for every CLV_Group for Income
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=customers,
    x='CLV_Group',
    y='Income',
    order=['Very Low','Low','Medium','High','Very High'],
    palette='Set3'
)
plt.title('Income Distribution by Customer Lifetime Value Group')
plt.xlabel('Customer Lifetime Value Group')
plt.ylabel('Income')
plt.show()

# %%
# Gender distribution per CLV_Group
plt.figure(figsize=(12, 8))
sns.countplot(
    data=customers,
    x='CLV_Group',
    hue='Gender',
    palette='Set2'
)
plt.title('Gender Distribution by Customer Lifetime Value Group')
plt.xlabel('Customer Lifetime Value Group')
plt.ylabel('Number of Customers')
plt.legend(title='Gender')
plt.show()


# %%
# Location Code distribution per CLV_Group
plt.figure(figsize=(12, 6))
sns.countplot(
    data=customers,
    x='CLV_Group',
    hue='Location Code',
    order=['Very Low','Low','Medium','High','Very High']
)
plt.title('Location Code Distribution by Customer Lifetime Value Group')
plt.xlabel('Customer Lifetime Value Group')
plt.ylabel('Number of Customers')
plt.legend(title='Location Code', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# %%
# Education distribution per CLV_Group
plt.figure(figsize=(12, 8))
sns.countplot(
    data=customers,
    x='CLV_Group',
    hue='Education',
    palette='Set1'
)
plt.title('Education Distribution by Customer Lifetime Value Group')
plt.xlabel('Customer Lifetime Value Group')
plt.ylabel('Number of Customers')
plt.legend(title='Education')
plt.show()


# %%
# Marital Status distribution per CLV_Group
plt.figure(figsize=(12, 8))
sns.countplot(
    data=customers,
    x='CLV_Group',
    hue='Marital Status',
    palette='Set3'
)
plt.title('Marital Status Distribution by Customer Lifetime Value Group')
plt.xlabel('Customer Lifetime Value Group')
plt.ylabel('Number of Customers')
plt.legend(title='Marital Status')
plt.show()


# %%
# Loyalty Status distribution per CLV_Group as a stacked bar chart 

plt.figure(figsize=(12, 8))
clv_loyalty_counts = customers.groupby(['CLV_Group', 'LoyaltyStatus']).size().unstack(fill_value=0)
clv_loyalty_counts.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Loyalty Status Distribution by Customer Lifetime Value Group')
plt.xlabel('Customer Lifetime Value Group')
plt.ylabel('Number of Customers')
plt.legend(title='Loyalty Status')
plt.show()

# %%
# Customer Tenure Days distribution per CLV_Group
# Filter out negative tenure days if any only for this plot
tenure_customers = customers[customers['CustomerTenureDays'] >= 0]
plt.figure(figsize=(12, 8))
sns.boxplot(
    data=tenure_customers,
    x='CLV_Group',
    y='CustomerTenureDays',
    palette='Set3'
)
plt.title('Customer Tenure Days Distribution by CLV Group')
plt.xlabel('CLV Group')
plt.ylabel('Tenure Days')
plt.show()

# %%
# Enrollment Type distribution per CLV_Group
plt.figure(figsize=(12, 8))
sns.countplot(  
    data=customers,
    x='CLV_Group',
    hue='EnrollmentType',
    palette='Set1'
)
plt.title('Enrollment Type Distribution by Customer Lifetime Value Group')
plt.xlabel('Customer Lifetime Value Group')
plt.ylabel('Number of Customers')
plt.legend(title='Enrollment Type')
plt.show()  

# %% [markdown]
# ## 5.1.3 Income Groups vs CLV Groups <a class="anchor" id="Customer Lifetime Value"></a>
# 
# 

# %% [markdown]
# We compare the two groups we defined before to find some insights.

# %%
# CLV_Group und Income Group Bar Chart
plt.figure(figsize=(10, 6))
sns.countplot(
    data=customers,
    x='CLV_Group',
    hue='Income_Group',
    order=['Very Low', 'Low', 'Medium', 'High', 'Very High']
)
plt.title('Customer Counts by CLV Group and Income Group')
plt.xlabel('CLV Group')
plt.ylabel('Number of Customers')
plt.legend(title='Income Group')
plt.show()

# %%
# Crosstable of CLV_Group and Income_Group
clv_income_crosstab = pd.crosstab(customers['CLV_Group'], customers['Income_Group'], normalize='index') * 100

# Visualisation as Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
    clv_income_crosstab,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu",
    cbar_kws={"label": "Percentage (%)"}
)
plt.title("CLV Group vs Loyalty Status Heatmap", fontsize=14, pad=15)
plt.xlabel("Loyalty Status")
plt.ylabel("CLV Group")
plt.show()

# Visualisation as as Bar Chart
clv_income_crosstab.plot(kind='bar', figsize=(10, 6))
plt.title("CLV Group vs Income Group Bar Chart", fontsize=14, pad=15)
plt.xlabel("CLV Group")
plt.ylabel("Percentage (%)")
plt.legend(title='Income Group')
plt.show()

# %% [markdown]
# ## 5.2 Behavioral Segmentation    <a class="anchor" id="Flights-EDA"></a>
# 

# %% [markdown]
# In this Section we look into some insights about the behaviour. This for example includes behaviour of the customers in the months. 

# %% [markdown]
# ### 5.2.1 Seasonal Analysis <a class="anchor" id="Customer Lifetime Value"></a>
# 

# %% [markdown]
# Some text as explanation

# %%
# Count the NumFlights per Month for each year 
monthly_flight_counts = flights.groupby(['Year', 'Month'])['NumFlights'].sum().reset_index()

# Plot the monthly flight counts for each year
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=monthly_flight_counts,
    x='Month',
    y='NumFlights',
    hue='Year',
    marker='o'
)
plt.title('Monthly Flight Counts by Year')
plt.xlabel('Month')
plt.ylabel('Total Number of Flights')
plt.xticks(range(1, 13))
plt.legend(title='Year')
plt.show()


# %%
# Count the NumFlights per Month for each year 
monthly_flight_counts = (
    flights.groupby(['Year', 'Month'])['NumFlights']
    .sum()
    .reset_index()
    .sort_values(['Year', 'Month'])
)

# --- Plot: Grouped Bar Chart ---
plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")

sns.barplot(
    data=monthly_flight_counts,
    x='Month',
    y='NumFlights',
    hue='Year',
    palette='viridis'
)


plt.title('Monthly Flight Counts by Year', fontsize=16, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Total Number of Flights')
plt.xticks(range(0, 12), [str(i+1) for i in range(12)])  # months 1–12
plt.legend(title='Year', loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()


# %%
# Count the NumFlights per Season for each year
seasonal_flight_counts = (
    flights.groupby(['Year', 'Season'])['NumFlights']
    .sum()
    .reset_index()
    .sort_values(['Year', 'Season'])
)

# Define season order (optional, to keep logical order)
season_order = ['Winter', 'Spring', 'Summer', 'Autumn']

# --- Plot: Grouped Bar Chart by Season ---
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

sns.barplot(
    data=seasonal_flight_counts,
    x='Season',
    y='NumFlights',
    hue='Year',
    palette='viridis',
    order=season_order
)

# Add small value labels on top of bars
for container in plt.gca().containers:
    plt.bar_label(container, fmt='%d', label_type='edge', fontsize=8, padding=2)

plt.title('Seasonal Flight Counts by Year', fontsize=16, fontweight='bold')
plt.xlabel('Season')
plt.ylabel('Total Number of Flights')
plt.legend(title='Year', loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()


# %%
# Need to be fixed

df_loyalty = flight_aggs.copy()
df_loyalty['LoyaltyStatus'] = df_loyalty['LoyaltyStatus'].str.lower()

# Filter for the specified loyalty statuses
df_loyalty = df_loyalty[df_loyalty['LoyaltyStatus'].isin(['star', 'aurora', 'nova'])]

# Check if the filtered DataFrame is empty before plotting
if df_loyalty.empty:
    print("Error: The filtered DataFrame is empty. Ensure 'star', 'aurora', or 'nova' (in any case) exist in your LoyaltyStatus column.")
else:
    # Calculate total flights per season and loyalty status
    df_season_loyalty = df_loyalty.groupby(['LoyaltyStatus', 'Season'])['NumFlights'].sum().unstack(fill_value=0)

    # Normalize to proportions (percentage of each loyalty group's total flights)
    df_season_loyalty_prop = df_season_loyalty.div(df_season_loyalty.sum(axis=1), axis=0)

    # Reorder seasons for consistent plotting
    df_season_loyalty_prop = df_season_loyalty_prop.reindex(columns=season_order)

    # Plotting - Stacked Bar Chart for Proportions
    plt.figure(figsize=(9, 6))
    df_season_loyalty_prop.plot(
        kind='bar',
        stacked=True,
        ax=plt.gca(), # Pass current axes to pandas plot
        color={'Winter': 'teal', 'Spring': 'green', 'Summer': 'red', 'Autumn': 'orange'}
    )

    plt.title('Distribution of Travel Seasons by Loyalty Status')
    plt.xlabel('Loyalty Status')
    plt.ylabel('Proportion of Total Flights (per Loyalty Group)')
    plt.xticks(rotation=0)
    plt.legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# %%
customers['SoloFlights'] = customers['total_flights'] - customers['total_flights_with_companions']

# 2. Aggregate by Loyalty Status. Filter the status names by converting to lower case.
df_solo_companion = customers.copy()
df_solo_companion['LoyaltyStatus'] = df_solo_companion['LoyaltyStatus'].str.lower()
df_solo_companion = df_solo_companion[df_solo_companion['LoyaltyStatus'].isin(['star', 'aurora', 'nova'])]

df_solo_companion = df_solo_companion.groupby('LoyaltyStatus')[['SoloFlights', 'total_flights_with_companions']].sum().reset_index()

# Rename columns for clarity in the plot legend
df_solo_companion = df_solo_companion.rename(columns={
    'total_flights_with_companions': 'With Companions',
    'SoloFlights': 'Solo Travel'
})

# Melt the DataFrame for Seaborn's barplot
df_melted = df_solo_companion.melt(
    id_vars='LoyaltyStatus',
    var_name='Travel Type',
    value_name='Total Flights'
)
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_melted,
    x='LoyaltyStatus',
    y='Total Flights',
    hue='Travel Type',
    palette={'Solo Travel': 'skyblue', 'With Companions': 'salmon'}
)

plt.title('Solo vs. Companion Travel by Loyalty Status (Total Flights)')
plt.xlabel('Loyalty Status')
plt.ylabel('Total Number of Flights')
plt.grid(axis='y', alpha=0.5)
plt.legend(title='Travel Type')
plt.show()

# %% [markdown]
# ## 5.3 Demographic Segmentation    <a class="anchor" id="Flights-EDA"></a>
# 

# %% [markdown]
# Some insights about the demographic segmentation of the customers.

# %% [markdown]
# ## 5.3.1 Proveince Analysis <a class="anchor" id="Customer Lifetime Value"></a>
# 

# %% [markdown]
# Province and Gender Distribution

# %%
plt.figure(figsize=(10,6))
sns.countplot(data=customers, x='Province or State', hue='Gender')
plt.title('Gender Distribution by Province')
plt.xlabel('Province')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# Province and Ecucation Distribution

# %%
plt.figure(figsize=(10,6))
sns.countplot(data=customers, x='Province or State', hue='Education')
plt.title('Education Distribution by Province')
plt.xlabel('Province')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
province_pct = (
    customers['Province or State']
    .value_counts(normalize=True) * 100
).round(2)

display(province_pct)

# %% [markdown]
# #### Province and Income (sum of total income by customers) Distribution

# %%
income_sum = (
    customers.groupby('Province or State')['Income']
    .sum()
    .reset_index(name='TotalIncome')
    .sort_values('TotalIncome', ascending=False)
)

fig = px.bar(
    income_sum,
    x='Province or State',
    y='TotalIncome',
    title='Total Income by Province (Sum)',
    text_auto='.2s',
    color='TotalIncome',
    color_continuous_scale='Blues'
)
fig.update_layout(xaxis_title='Province', yaxis_title='Total Income', height=500)
fig.show()

# %% [markdown]
# #### Province and Income (mean of all customer) Distribution

# %%
income_mean = (
    customers.groupby('Province or State')['Income']
    .mean()
    .reset_index(name='AverageIncome')
    .sort_values('AverageIncome', ascending=False)
)

fig = px.bar(
    income_mean,
    x='Province or State',
    y='AverageIncome',
    title='Average Income per Customer by Province',
    text_auto='.2f',
    color='AverageIncome',
    color_continuous_scale='Viridis'
)

# Add thousand separators to y-axis and hover labels
fig.update_layout(
    xaxis_title='Province',
    yaxis_title='Average Income',
    height=500,
    yaxis_tickformat=',',          # ← adds thousand separators
)

# Format hover labels and text with commas
fig.update_traces(
    hovertemplate='Province: %{x}<br>Average Income: %{y:,.0f}',
    texttemplate='%{y:,.0f}'      # ← show commas in bar text labels
)

fig.show()


# %% [markdown]
# #### Province and Marital Status Distribution

# %% [markdown]
# Maybe we should do the percentage here instead of the counts

# %%
plt.figure(figsize=(10,6))
sns.countplot(data=customers, x='Province or State', hue='Marital Status')
plt.title('Marital Status by Province')
plt.xlabel('Province')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Province and LoyaltyStatus Distribution

# %%
plt.figure(figsize=(10,6))
sns.countplot(data=customers, x='Province or State', hue='LoyaltyStatus')
plt.title('Loyalty Status by Province')
plt.xlabel('Province')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Province and Customer LifeTime Value (sum) Distribution

# %%
clv_sum = (
    customers.groupby('Province or State')['Customer Lifetime Value']
    .sum()
    .reset_index(name='TotalCLV')
    .sort_values('TotalCLV', ascending=False)
)

fig = px.bar(
    clv_sum,
    x='Province or State',
    y='TotalCLV',
    title='Total Customer Lifetime Value by Province (Sum)',
    text_auto='.2s',
    color='TotalCLV',
    color_continuous_scale='Purples'
)
fig.update_layout(xaxis_title='Province', yaxis_title='Total CLV', height=500)
fig.show()

# %% [markdown]
# #### Province and Customer LifeTime Value (Mean) Distribution

# %%
total_clv = customers['Customer Lifetime Value'].sum()
print(f"Total Customer Lifetime Value: {total_clv:,.2f}")

# %%
clv_by_province = (
    customers.groupby('Province or State')['Customer Lifetime Value']
    .sum()
    .reset_index()
    .rename(columns={'Customer Lifetime Value': 'Total_CLV'})
    .sort_values('Total_CLV', ascending=False)
)

display(clv_by_province)

# %%
clv_by_province['Share_%'] = (
    100 * clv_by_province['Total_CLV'] / total_clv
).round(2)

display(clv_by_province)

# %%
clv_mean = (
    customers.groupby('Province or State')['Customer Lifetime Value']
    .mean()
    .reset_index(name='AverageCLV')
    .sort_values('AverageCLV', ascending=False)
)

fig = px.bar(
    clv_mean,
    x='Province or State',
    y='AverageCLV',
    title='Average Customer Lifetime Value per Customer by Province',
    text_auto='.2f',
    color='AverageCLV',
    color_continuous_scale='Magma'
)
fig.update_layout(xaxis_title='Province', yaxis_title='Average CLV', height=500)
fig.show()

# %% [markdown]
# #### Province and IsActive Distribution

# %%
active_counts = (
    customers.groupby(['Province or State', 'IsActive'])
    .size()
    .reset_index(name='CustomerCount')
)

fig = px.bar(
    active_counts,
    x='Province or State',
    y='CustomerCount',
    color='IsActive',
    barmode='group',
    title='Active vs Inactive Customers by Province'
)
fig.update_layout(xaxis_title='Province', yaxis_title='Number of Customers', height=500)
fig.show()


# %% [markdown]
# #### Province and Total Flights Distribution

# %%
flights_sum = (
    customers.groupby('Province or State')['total_flights']
    .sum()
    .reset_index(name='TotalFlights')
    .sort_values('TotalFlights', ascending=False)
)

fig = px.bar(
    flights_sum,
    x='Province or State',
    y='TotalFlights',
    title='Total Flights by Province',
    text_auto='.0f',
    color='TotalFlights',
    color_continuous_scale='Tealgrn'
)
fig.update_layout(xaxis_title='Province', yaxis_title='Total Flights', height=500)
fig.show()

# %% [markdown]
# #### Province and Total Distance Distribution

# %%
distance_sum = (
    customers.groupby('Province or State')['total_distance']
    .sum()
    .reset_index(name='TotalDistance')
    .sort_values('TotalDistance', ascending=False)
)

fig = px.bar(
    distance_sum,
    x='Province or State',
    y='TotalDistance',
    title='Total Distance by Province',
    text_auto='.0f',
    color='TotalDistance',
    color_continuous_scale='Cividis'
)
fig.update_layout(xaxis_title='Province', yaxis_title='Total Distance (km)', height=500)
fig.show()


# %% [markdown]
# ### 5.3.2 City Analysis <a class="anchor" id="Customer Lifetime Value"></a>
# 

# %% [markdown]
# #### City and Province distribution

# %%
top_cities_by_province = (
    customers.groupby(["Province or State", "City"])
             .size()
             .rename("CustomerCount")
             .reset_index()
             .sort_values(["Province or State", "CustomerCount"], ascending=[True, False])
             .groupby("Province or State")
             .head(5)
)

# (Opcional) percentagem dentro da province
totals = top_cities_by_province.groupby("Province or State")["CustomerCount"].transform("sum")
top_cities_by_province["ShareWithinTopN"] = 100 * top_cities_by_province["CustomerCount"] / totals

# Mostrar
try:
    df_show = top_cities_by_province.copy()
    df_show["CustomerCount"] = df_show["CustomerCount"].map("{:,}".format)
    df_show["ShareWithinTopN"] = df_show["ShareWithinTopN"].map("{:.1f}%".format)
    display(df_show)
except NameError:
    print(top_cities_by_province)

# %% [markdown]
# #### City and Income (sum) Distribution

# %%
# --- 1) Prepare data
df = customers.copy()
df["Income"] = pd.to_numeric(df["Income"], errors="coerce")
df = df.dropna(subset=["Income", "City"])

# --- 2) Total income per city
city_sum = (
    df.groupby("City", as_index=False)["Income"]
      .sum()
      .sort_values("Income", ascending=False)
      .head(10)
)

# --- 3) Plot
fig = px.bar(
    city_sum,
    x="City",
    y="Income",
    title="Top 10 Cities by Total Income",
    text_auto=".0f",
    color="Income",
    color_continuous_scale="Cividis"
)

# Format with thousand separators
fig.update_traces(texttemplate="%{y:,.0f}")
fig.update_layout(
    xaxis_title="City",
    yaxis_title="Total Income",
    yaxis_tickformat=",",
    height=500
)

fig.show()

# %% [markdown]
# #### City and Income (average) Distribution

# %%
# --- 1) Prepare data
df = customers.copy()
df["Income"] = pd.to_numeric(df["Income"], errors="coerce")
df = df.dropna(subset=["Income", "City"])

# --- 2) Average income per city
city_mean = (
    df.groupby("City", as_index=False)["Income"]
      .mean()
      .sort_values("Income", ascending=False)
      .head(10)
)

# --- 3) Plot
fig = px.bar(
    city_mean,
    x="City",
    y="Income",
    title="Top 10 Cities by Average Income",
    text_auto=".0f",
    color="Income",
    color_continuous_scale="Viridis"
)

# Format with thousand separators
fig.update_traces(texttemplate="%{y:,.0f}")
fig.update_layout(
    xaxis_title="City",
    yaxis_title="Average Income",
    yaxis_tickformat=",",
    height=500
)

fig.show()


# %% [markdown]
# #### City and Gender Disttibution

# %%
plt.figure(figsize=(10,6))
sns.countplot(data=customers, x='City', hue='Gender')
plt.title('Gender Distribution by Province')
plt.xlabel('City')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# #### City and Customer LifeTime Value distribution

# %%
df = customers.copy()
df["Customer Lifetime Value"] = pd.to_numeric(df["Customer Lifetime Value"], errors="coerce")
df = df.dropna(subset=["City", "Customer Lifetime Value"])

# Average per city
clv_city = (
    df.groupby("City", as_index=False)["Customer Lifetime Value"]
      .mean()
      .sort_values("Customer Lifetime Value", ascending=False)
      .head(10)
)

plt.figure(figsize=(10, 5))
sns.barplot(
    data=clv_city,
    x="Customer Lifetime Value",
    y="City",
    palette="viridis"
)
plt.title("Top 10 Cities by Average Customer Lifetime Value", fontsize=14)
plt.xlabel("Average CLV")
plt.ylabel("City")
plt.tight_layout()
plt.show()

# %% [markdown]
# #### City and Customer Tenure Days distribution

# %%
df = customers.copy()
df["Customer Lifetime Value"] = pd.to_numeric(df["Customer Lifetime Value"], errors="coerce")
df = df.dropna(subset=["City", "Customer Lifetime Value"])

# Average per city
clv_city = (
    df.groupby("City", as_index=False)["Customer Lifetime Value"]
      .mean()
      .sort_values("Customer Lifetime Value", ascending=False)
      .head(10)
)

plt.figure(figsize=(10, 5))
sns.barplot(
    data=clv_city,
    x="Customer Lifetime Value",
    y="City",
    palette="viridis"
)
plt.title("Top 10 Cities by Average Customer Lifetime Value", fontsize=14)
plt.xlabel("Average CLV")
plt.ylabel("City")
plt.tight_layout()
plt.show()

# %% [markdown]
# #### City and TotalFlights distribution

# %%
df = customers.copy()
df["total_flights"] = pd.to_numeric(df["total_flights"], errors="coerce")
df = df.dropna(subset=["City", "total_flights"])

flights_city = (
    df.groupby("City", as_index=False)["total_flights"]
      .mean()
      .sort_values("total_flights", ascending=False)
      .head(10)
)

plt.figure(figsize=(10, 5))
sns.barplot(
    data=flights_city,
    x="total_flights",
    y="City",
    palette="rocket"
)
plt.title("Top 10 Cities by Average Total Flights", fontsize=14)
plt.xlabel("Average Total Flights")
plt.ylabel("City")
plt.tight_layout()
plt.show()


# %% [markdown]
# #### City vs Total Distance

# %%
df = customers.copy()
df["total_distance"] = pd.to_numeric(df["total_distance"], errors="coerce")
df = df.dropna(subset=["City", "total_distance"])

distance_city = (
    df.groupby("City", as_index=False)["total_distance"]
      .mean()
      .sort_values("total_distance", ascending=False)
      .head(10)
)

plt.figure(figsize=(10, 5))
sns.barplot(
    data=distance_city,
    x="total_distance",
    y="City",
    palette="flare"
)
plt.title("Top 10 Cities by Average Total Distance", fontsize=14)
plt.xlabel("Average Total Distance")
plt.ylabel("City")
plt.tight_layout()
plt.show()


# %% [markdown]
# # 6. Conclusion <a class="anchor" id="conclusion"></a>


