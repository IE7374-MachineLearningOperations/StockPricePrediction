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

This notebook provides a comprehensive overview of the data schema and statistics:
- **Descriptive Statistics**: Calculates statistics such as mean, median, standard deviation, etc., for each feature.
- **Schema Consistency Checks**: Validates that the data structure is consistent across different datasets.
- **Anomaly Detection**: Identifies unusual or out-of-bound values that may need correction.
- **Feature Distributions**: Visualizes the distribution of each feature to ensure normality and detect skewness or other issues.

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
