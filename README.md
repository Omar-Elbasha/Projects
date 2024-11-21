# KAGGLE Competition: House Prices - Advanced Regression Techniques

#### Video Demo:  https://youtu.be/uybwogYdSuI

#### Description:
Project consists of a solution to House Pricing Advanced Regression problem on Kaggle using the Ames Housing Dataset.  
[View the competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)

---

### Dataset shape:
Training data consists of 81 columns and 1460 rows.  
Of the 81 columns, 79 are features describing different details of each house, one is arbitrary Id and the label 'SalePrice' is the amount the house was sold for.

---

### Dataset inspection:
- Many columns had more than 50% missing values.
- For the features: there were categorical (nominal and ordinal) columns, and numerical columns.
- For the label: the data was skewed, with some data points way larger than most.

---

### Feature Engineering:
- Used 80% of data for training and 20% of data for testing
- Removed columns that had a majority of missing values:'PoolQC', 'MiscFeature', 'Alley','GarageYrBlt', 'GarageCond','BsmtFinType2','Fence'])
- Separated categorical and nominal data.
- ordinal_columns = ['LotShape', 'LandContour','Utilities',
        'LandSlope',  'BsmtQual',  'BsmtFinType1',  'CentralAir',  'Functional',
        'FireplaceQu', 'GarageFinish', 'GarageQual', 'PavedDrive', 'ExterCond',
        'KitchenQual', 'BsmtExposure', 'HeatingQC','ExterQual', 'BsmtCond']
    
- nominal_columns = ['Street', 'LotConfig','Neighborhood',
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
        'RoofStyle', 'Exterior1st', 'Exterior2nd',
        'MasVnrType','Foundation',  'Electrical',
        'SaleType', 'MSZoning', 'SaleCondition',
        'Heating', 'GarageType', 'RoofMatl']
- Generated new features based on existing ones such as "TotalBathrooms" (sum of the number of full bathrooms + 1/2 of the number of half bathrooms), "TotalSF," and others, then dropped the original columns.


- Did log+1 scaling on the label to get a better normal distribution.

---

### Preprocessing:
- Used transformers and pipelines to prepare the data.
- **Categorical data**: Simple Imputation with most_frequent strategy.
- **Ordinal data**: OrdinalEncoding.
- **Nominal data**: OneHotEncoding.
- **Numerical data**: Simple Imputation with mean strategy.
- ordinal_transformer = Pipeline(steps=[
        ('impute', SimpleImputer(strategy = 'most_frequent')),
        ('ordinal_encode', OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1))
    ])

    nominal_transformer = Pipeline(steps=[
        ('impute', SimpleImputer(strategy = 'most_frequent')),
        ('onehot_encode', OneHotEncoder(handle_unknown = 'ignore', sparse_output = False))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy = 'median'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('ord', ordinal_transformer, ordinal_columns),
            ('nom', nominal_transformer, nominal_columns)
        ],
        remainder='passthrough'
    )

No scaling for numerical data (except for the label) because we are using RandomForest and XGBoost, which are indifferent to scaling of features.

---

### Model training:
- Tried both RandomForest and XGBoost models; XGBoost performed significantly better on its own.
- Later tried taking the average between them, but XGBoost alone still performed better.
- In the end, made the submission predictions using XGBoost only.
- Model paramters of RandomForest
    rfr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=13))
    ])

    param_grid_rfr = {
        'regressor__max_depth': [5, 10, 15],
        'regressor__n_estimators': [100, 250, 500],
        'regressor__min_samples_split': [3, 5, 10]
    }

    rfr_cv = GridSearchCV(
        rfr,
        param_grid_rfr,
        cv = 5,
        scoring = 'neg_mean_squared_error',
        n_jobs = -1
    )
    
- Model paramaters of XGBoost
    XGB = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=13))
    ])
    
    param_grid_XGB = {
        'regressor__learning_rate': [0.05, 0.1, 0.2],
        'regressor__n_estimators': [300],
        'regressor__max_depth': [3],
        'regressor__min_child_weight': [1,2,3],
        'regressor__gamma': [0, 0.1, 0.2],
        'regressor__subsample': [0.8, 0.9, 1.0],
        'regressor__colsample_bytree': [0.8, 0.9, 1.0],
    }
    
    print("Training XGBOOST model")
    xgb_cv = GridSearchCV(XGB, param_grid_XGB, cv = 3, scoring='neg_mean_squared_error', n_jobs = -1)
---

### End results:
**Leaderboard**: Top 25% of submissions - 1528th out of 6060 submissions with an RMSE of 0.13211.
