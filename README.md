# Bank Transaction Fraud Detection

A Jupyter Notebook project that explores bank transaction data, preprocesses features, and compares several machine-learning classifiers for fraud detection.

## Project workflow

1. Load a bank transactions CSV dataset.
2. Explore missing values, duplicates, numerical and categorical columns.
3. Prepare features and visualize the data.
4. Use feature importance and PCA for analysis.
5. Split the data into training and test sets; apply scaling and techniques for class imbalance.
6. Train and compare classification models, including Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, AdaBoost, Bagging, and KNN.
7. Evaluate classification performance with accuracy, classification reports, confusion matrices, and ROC curves.

## Files

- `Bank-Transaction-Fraud-Detection.ipynb` — the main notebook.
- `requirements.txt` — Python dependencies.
- `.gitignore` — patterns for local datasets, model artifacts, caches, and secrets that should not be committed.

## Run locally

1. Install Python 3 and Jupyter Notebook.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Obtain the appropriate transaction dataset from its authorized source and store it locally. **The dataset is not included in this repository.**
4. Open the notebook:

   ```bash
   jupyter notebook Bank-Transaction-Fraud-Detection.ipynb
   ```

5. **Before running the data-loading cell**, change the notebook's Kaggle-specific `/kaggle/input` directory scan to an explicit local CSV path, for example:

   ```python
   file_path = "data/bank_transactions.csv"  # replace with the real CSV filename
   df = pd.read_csv(file_path)
   ```

   The notebook currently discovers files under `/kaggle/input` and then reads the last discovered path. This will need to be adjusted for other environments.

## Results and limitations

The notebook describes approximately **95% overall classification accuracy**, but its own conclusion reports near-zero recall for the fraud class and ROC AUC near 0.5. High accuracy on an imbalanced dataset **does not mean fraud is being detected reliably**. This project is an experimental learning project and is **not suitable for production banking fraud decisions** without substantial validation and improvement.

## Data privacy

The data schema includes customer names, contact details, email addresses, and transaction identifiers. Do not publish actual customer records or output cells that reveal personal or financial information. Check notebook outputs before making the repository public.

## Attribution

If this notebook or its dataset was adapted from another creator or Kaggle source, include the original source URL and applicable credit/license here before publishing.
