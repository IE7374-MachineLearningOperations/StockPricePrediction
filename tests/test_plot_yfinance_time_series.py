import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath("pipeline/airflow/dags/src"))

# Import your function
from plot_yfinance_time_series import *

# Test functions will be defined here
