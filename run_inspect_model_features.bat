@echo off
cd /d c:\Users\LENOVO\shivansh\RAGABS~1
c:\Users\LENOVO\shivansh\RAGABS~1\.venv\Scripts\python.exe -c "from abstention_model.features import load_features, FEATURE_COLUMNS; print('FEATURE_COLUMNS:', FEATURE_COLUMNS); print(); X, y, meta = load_features('data/train_features.parquet'); print('X shape:', X.shape); print('X columns (if DataFrame):', list(X.columns) if hasattr(X, 'columns') else 'numpy array'); print(); print('First row feature values:'); print(X.iloc[0] if hasattr(X, 'iloc') else X[0])"
