# House Price Predictor

This project is a simple machine learning pipeline for predicting house prices using a dataset of demographic and economic features. It utilizes a Random Forest Regressor combined with preprocessing steps to handle both categorical and numerical variables.

## 📂 Project Structure

- `house_price_predictor.py`: Main script that loads the data, preprocesses it, trains the model, and evaluates the performance.
- `veriler.csv`: Dataset file containing the training data.

## 📊 Dataset

The dataset (`veriler.csv`) includes the following columns:

- `yas` (age): Numeric
- `maas` (salary): Numeric
- `deneyim` (experience): Numeric
- `sehir` (city): Categorical
- `ev_fiyati` (house price): Target variable, numeric

## ⚙️ How It Works

1. **Data Loading**: Loads data from a CSV file.
2. **Feature Selection**: Uses `yas`, `maas`, `deneyim`, and `sehir` as features.
3. **Preprocessing**:
   - Categorical column `sehir` is one-hot encoded.
   - Numerical columns are passed through as-is.
4. **Model**:
   - Uses `RandomForestRegressor` with 100 estimators.
5. **Training and Testing**:
   - Data is split into training and testing sets (67% / 33%).
   - The model is trained on the training data.
   - Predictions are made on the test set.
6. **Evaluation**:
   - Reports Mean Squared Error (MSE) and R² Score.

## 📈 Output

After running the script, the following metrics are printed:

- Mean Squared Error
- R² Score

These metrics help evaluate the model’s prediction accuracy.

## 🛠 Requirements

- Python 3.x
- pandas
- scikit-learn

Install dependencies using:

```bash
pip install pandas scikit-learn
