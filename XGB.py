import pandas as pd
import numpy as np
from sklearn.preprocessing import  StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy import stats
from xgboost import XGBRegressor

def main():

    df = load_data()
    df_train = df.drop(columns='Id')
    preprocessor = get_preprocessor(df)



    X = df_train.drop(columns = ['SalePrice'])
    y = df_train['SalePrice']
    
    X_train , X_test , y_train , y_test = train_test_split(X , y , random_state = 13 , test_size = 0.2)

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
    xgb_cv.fit(X_train, y_train)
    xgb_rmse = np.sqrt(-1 * xgb_cv.best_score_)
    print(f"XGB: RMSE on training set is {xgb_rmse}")
    
    # Predict on test set
    y_test_pred_xgb = xgb_cv.best_estimator_.predict(X_test)
    # Compute RMSE on the test set
    test_rmse_xgb = np.sqrt(mean_squared_error(y_test, y_test_pred_xgb))
    print(f"XGB: RMSE on test set is {test_rmse_xgb}")

    df_test = load_test()
    y_pred_xgb = xgb_cv.best_estimator_.predict(df_test.drop(columns="Id"))
    results = pd.DataFrame({
        'Id': df_test['Id'],
        'SalePrice': np.expm1(y_pred_xgb)
        })
    results.to_csv('submission.csv', index=False)


    return

def load_data():
    try:
        df = pd.read_csv('train.csv')
    except FileNotFoundError:
        print("The file 'train.csv' has not been found")
        return
    
    df = modify_features(df)
    return df


def load_test():
    try:
        df_test = pd.read_csv('test.csv')
    except FileNotFoundError:
        print("The file 'test.csv' has not been found")
        return
    
    df_test = modify_test_features(df_test)
    return df_test


def modify_features(df):

    df = df.drop(columns=[
        'PoolQC', 'MiscFeature', 'Alley',
        'GarageYrBlt', 'GarageCond','BsmtFinType2','Fence'])
    
    df['SalePrice'] = np.log1p(df['SalePrice'])
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['Remodeled'] = np.where(df['YearBuilt'] == df['YearRemodAdd'], 0, 1)
    df['TotalBathrooms'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])
    df = df.drop(columns = ['FullBath','HalfBath', 'BsmtFullBath', 'BsmtHalfBath'])
    df['LiveableSF'] = df['1stFlrSF'] + df['2ndFlrSF']
    df['BasementSF'] = df['BsmtFinSF1'] + df['BsmtFinSF2']
    df = df.drop(columns = ['1stFlrSF','2ndFlrSF','BsmtFinSF1','BsmtFinSF2'])
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
    df = df.drop(columns = ['OpenPorchSF','3SsnPorch','EnclosedPorch','ScreenPorch','WoodDeckSF'])

    return df

def modify_test_features(df):
    df = df.drop(columns=[
        'PoolQC', 'MiscFeature', 'Alley',
        'GarageYrBlt', 'GarageCond','BsmtFinType2','Fence'])
    
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['Remodeled'] = np.where(df['YearBuilt'] == df['YearRemodAdd'], 0, 1)
    df['TotalBathrooms'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])
    df = df.drop(columns = ['FullBath','HalfBath', 'BsmtFullBath', 'BsmtHalfBath'])
    df['LiveableSF'] = df['1stFlrSF'] + df['2ndFlrSF']
    df['BasementSF'] = df['BsmtFinSF1'] + df['BsmtFinSF2']
    df = df.drop(columns = ['1stFlrSF','2ndFlrSF','BsmtFinSF1','BsmtFinSF2'])
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
    df = df.drop(columns = ['OpenPorchSF','3SsnPorch','EnclosedPorch','ScreenPorch','WoodDeckSF'])

    return df

def get_preprocessor(df):

    # Choose categorical and numerical columns
    ordinal_columns = ['LotShape', 'LandContour','Utilities',
        'LandSlope',  'BsmtQual',  'BsmtFinType1',  'CentralAir',  'Functional',
        'FireplaceQu', 'GarageFinish', 'GarageQual', 'PavedDrive', 'ExterCond',
        'KitchenQual', 'BsmtExposure', 'HeatingQC','ExterQual', 'BsmtCond']
    
    nominal_columns = ['Street', 'LotConfig','Neighborhood',
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
        'RoofStyle', 'Exterior1st', 'Exterior2nd',
        'MasVnrType','Foundation',  'Electrical',
        'SaleType', 'MSZoning', 'SaleCondition',
        'Heating', 'GarageType', 'RoofMatl']
    
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    #numerical_columns = numerical_columns.drop('SalePrice')
    numerical_columns = numerical_columns.drop('SalePrice', errors='ignore')
    numerical_columns = numerical_columns.drop('Id', errors='ignore')


    ordinal_transformer = Pipeline(steps=[
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

    return preprocessor



if __name__ == "__main__":
    main()