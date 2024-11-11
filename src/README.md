# Data Pipeline Assignment Phase
## Notebooks and Key Steps

### 1. `PROJECT_DATA_CLEANING.ipynb`

This notebook handles data cleaning and includes the following steps:
- **Library Imports**: Uses libraries like Pandas, Numpy, Matplotlib, and Seaborn for data manipulation and visualization.
- **Data Loading**: Loads data from `final_dataset.csv`.
- **Missing Value Imputation**: Implements `SimpleImputer` to handle missing values in the dataset.
- **Outlier Detection and Removal**: Applies methods to detect and remove outliers to improve data quality.
- **Feature Selection**: Uses `SelectKBest`, `f_regression`, and Recursive Feature Elimination (RFE) with `RandomForestRegressor` for selecting important features.
- **Scaling**: Standardizes features using `StandardScaler` for improved model performance.

### 2. `data_preprocessing.ipynb`

This notebook is focused on preparing data for further analysis:
- **Data Type Conversion**: Converts `date` columns to datetime and adjusts numerical columns to appropriate types.
- **Encoding**: Handles encoding of categorical variables as necessary for model compatibility.
- **Scaling and Normalization**: Ensures that numerical features are scaled to improve model accuracy and convergence.
- **Visualization**: Provides data insights with visualizations to understand distributions and detect potential issues.

### 3. `DataSchema_Stats.ipynb`
Here's a breakdown of README on validating data quality, including both a general overview and specific details related to schema and statistics generation.
---

### Data Validation: Schema and Statistics Generation

Data quality validation ensures that data aligns with expected standards before processing or analysis. This process helps identify and mitigate issues early, enabling cleaner datasets and more reliable insights. The validation process primarily involves **schema validation** and **statistics generation**.

---

#### General Overview

1. **Schema Validation**: Ensures that the dataset's structure (columns, data types, and constraints) matches predefined standards. This step verifies column names, data types, allowable value ranges, nullability, and unique constraints.

2. **Statistics Generation**: Provides a comprehensive summary of each dataset's characteristics, such as mean, median, standard deviation, and distribution of values. These metrics reveal potential anomalies, like outliers or unexpected distributions, which could indicate data issues.

3. **Data Quality Checks**: Regularly generating statistics and performing schema checks ensures that data quality remains consistent over time, especially when working with evolving data sources.

---

#### Specific Details

1. **Schema Validation Process**:
   - **Define Expected Schema**: Specify structure of our dataset, including column names, data types (e.g., integer, float, string), and constraints (e.g., non-null, unique).
   - **Validate Data Types**: Ensured each column’s data type aligns with the expected schema. Mismatched data types (e.g., string data in a numeric column) are flagged as issues.
   - **Check Null Values and Constraints**: Validate the presence or absence of null values based on predefined rules. For example, critical columns marked as non-null must not contain any missing values.
   - **Identify Unique Violations**: If columns are expected to contain unique values (e.g., IDs), check for duplicates. Duplicate records often indicate data integrity issues.

2. **Statistics Generation Process**:
   - **Descriptive Statistics**: Common metrics like mean, median, mode, minimum, maximum, and standard deviation. These measures help evaluate whether values are within an acceptable range.
   - **Frequency and Distribution Analysis**: Assess the frequency of categorical values and the distribution of numerical values. Unusual patterns, such as skewed distributions or unexpected categories, may indicate data issues.
   - **Outlier Detection**: Identify outliers, which could result from data entry errors or represent exceptional cases that need separate handling.
   - **Completeness and Consistency Checks**: Measure the proportion of missing data in each column, and assess consistency across.

3. **Anomalies Detection**

    - **Missing Value Analysis**: Identifies any fields with missing data. Depending on the importance and the context of the field, the missing values might be filled using statistical methods (e.g., mean, median) or flagged for further inspection.

    - **Outlier Detection**: Employs statistical methods to detect outliers within numerical data. Calculating z-scores or leveraging the Interquartile Range (IQR) to find values that deviate significantly from the central tendency of the data, suggesting possible anomalies.

---

1. **Data Completeness**: Ensure all required columns are present and contain no null values (or as expected).
2. **Data Consistency**: Use consistent data types and expected value ranges.
3. **Statistical Expectations**: Ensure statistical summaries align with domain knowledge (e.g., average values fall within expected ranges).
4. **Outlier Management**: Define thresholds for acceptable data points and set rules for handling extreme values.

### 4. `Feature Engineering.ipynb`

This notebook is dedicated to generating new features and selecting relevant ones:
- **Interaction Terms**: Creates interaction terms between features to capture additional predictive signals.
- **Dimensionality Reduction**: Uses Principal Component Analysis (PCA) to reduce dimensionality where necessary.
- **Domain-Specific Feature Creation**: Extracts meaningful features based on domain knowledge (e.g., customer segmentation features like RFM).
- **Feature Selection**: Applies correlation analysis and feature importance metrics to retain the most informative features.

## How to Run the Project

1. **Clone the Repository**
   ```bash
   git clone https://github.com/IE7374-MachineLearningOperations/StockPricePrediction.git
   cd Stock-Price-Prediction/src/
   ```
2. **Execute the Notebooks**
   Run the notebooks for understanding results:
   - `PROJECT_DATA_CLEANING.ipynb`: Clean and prepare the raw data.
   - `data_preprocessing.ipynb`: Preprocess the data for further analysis.
   - `DataSchema_Stats.ipynb`: Analyze the schema and statistics of the data.
   - `Feature Engineering.ipynb`: Generate and select final features for modeling.
