# Asteroid Diameter Prediction with Deep Neural Networks

## Overview
This notebook demonstrates the process of building and evaluating Deep Neural Network (DNN) models to predict the diameter of asteroids based on various orbital and physical characteristics. The project covers data loading, extensive Exploratory Data Analysis (EDA), robust preprocessing using `ColumnTransformer` and pipelines, and the development of multiple DNN models with hyperparameter tuning (learning rate, dropout) and best practices like Early Stopping and Model Checkpointing.

## Dataset
The dataset used for this project is `dataset.csv`, containing information about a large number of asteroids. Key features include orbital elements (e.g., semi-major axis, eccentricity, inclination), physical properties (e.g., absolute magnitude 'H', albedo), and the target variable, 'diameter' (in km).

## Project Structure

1.  **Libraries:** Import necessary libraries for data manipulation, visualization, and deep learning (`pandas`, `numpy`, `matplotlib`, `seaborn`, `tensorflow`, `scikit-learn`).
2.  **Data Loading:** Load the `dataset.csv` file into a pandas DataFrame.
3.  **Exploratory Data Analysis (EDA):**
    *   Descriptive statistics (`.describe()`, `.info()`).
    *   Missing value analysis (`.isna().sum()`) and duplicate check (`.duplicated().sum()`).
    *   Correlation heatmap to identify relationships between numerical features and the target variable (`diameter`).
    *   Distribution plots (histograms, box plots) for `diameter` and `H` (absolute magnitude).
    *   Insights into how orbital characteristics might influence asteroid size.
4.  **Data Preprocessing Pipeline:**
    *   **Handling Missing Values:** Rows with missing `diameter` values are dropped. Other missing values are handled via imputation within the pipeline.
    *   **Feature Selection:** Irrelevant columns (identifiers, highly missing, uncertainty estimates, epoch-related) are dropped.
    *   **Binary Encoding:** 'neo' and 'pha' columns (Near-Earth Object, Potentially Hazardous Asteroid) are manually encoded to 0/1.
    *   **`ColumnTransformer`:** A robust preprocessing pipeline is built using `scikit-learn`'s `ColumnTransformer`:
        *   **Numerical Features:** Imputed with the median and scaled using `StandardScaler`.
        *   **Categorical Features:** Imputed with the most frequent value and one-hot encoded (`class` feature).
    *   **Train-Test Split:** The preprocessed data is split into training and testing sets (`X_train_simplified`, `X_test_simplified`, `y_train_simplified`, `y_test_simplified`).
5.  **Deep Neural Network Model Development:**
    *   **Input Shape:** Dynamically determined from the preprocessed data.
    *   **Baseline Model:** A sequential DNN with `Dense` layers and `ReLU` activation, outputting a single `linear` unit for regression. Compiled with `Adam` optimizer (learning rate 0.001) and `mean_squared_error` loss.
    *   **Hyperparameter Tuning - Learning Rate:** A second model with a reduced learning rate (0.0001) to observe its effect on convergence and performance.
    *   **Hyperparameter Tuning - Dropout:** A third model incorporating `Dropout` layers to mitigate overfitting.
    *   **Best Practices:** All models use `EarlyStopping` (monitoring `val_loss`, patience=10, `restore_best_weights=True`) and `ModelCheckpoint` (saving the best model based on `val_loss`) during training.
6.  **Model Evaluation & Visualization:**
    *   A custom function `evaluate_model_and_plot` is defined to:
        *   Load the best saved model (`.keras` format).
        *   Evaluate on the unseen test set, reporting MAE, MSE, RMSE, and R2 score.
        *   Plot Actual vs. Predicted values.
        *   Plot Residual Error Distribution and Residuals vs. Predicted Values to identify patterns.
        *   Identify and display the top 10 largest prediction errors.
    *   Each of the three developed models (Baseline, LR Tuned, Dropout) is evaluated using this function.
7.  **Key Questions and Insights:**
    *   Discussion on the most suitable evaluation metrics (RMSE, R2 score) for this regression problem.
    *   Analysis of patterns observed in residual plots (e.g., heteroscedasticity).
    *   Identification of asteroid types most difficult to predict accurately (e.g., larger asteroids).

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd asteroid-diameter-prediction
    ```
2.  **Install dependencies:**
    Ensure you have Python 3.x installed. Install the required libraries:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
    ```
3.  **Download the dataset:**
    Place the `dataset.csv` file directory or update the `file_path` variable in the notebook to point to the correct location of your dataset.
4.  **Execute the notebook:**
    Open and run the notebook cells sequentially in a Jupyter environment (e.g., Google Colab, Jupyter Lab, Jupyter Notebook).
