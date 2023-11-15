
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge

f'DHCP_test_nn-UNet_extracted_features.csv'

def prepare_data(df, gender, variable):
    if gender in ["Male", "Female"]:
        df = df.query('gender==@gender')
    X = []
    for i,row in df.iterrows():
        X.append(list(eval(row[f'seg87_surface_to_volume_ratio']).values()) + list(eval(row[f'seg87_relational_volume']).values()))
    X = np.array(X)
    y = np.array(df[variable].values)

    return X, y

def pearson_correlation(X, y):
    correlations = []
    for x in X.transpose():
        correlations.append(np.abs(pearsonr(x,y)[0]))
    return np.array(correlations)

def full_regression(X, gt, X_test, gt_test, regressor):
    reg_scores = []
    reg_mses = []
    reg_maes = []
    y = gt - np.mean(gt)
    y_test = gt_test - np.mean(gt)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)
    model = regressor.fit(X_scaled, y)
    reg_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    reg_mse = mse(y_test, y_pred)
    reg_mae = mae(y_test, y_pred)

    reg_scores.append(reg_score)
    reg_mses.append(reg_mse)
    reg_maes.append(reg_mae)

    reg_score = np.mean(reg_scores)
    reg_rmse = mse(y_test, y_pred, squared=False)
    reg_mae_std = np.std(np.abs(y_test-y_pred))
    reg_mae = np.mean(np.abs(y_test-y_pred))
    return regressor, scaler, reg_score, reg_rmse, reg_mae, reg_mae_std, np.mean(gt)


def most_correlated_regression(X, gt, X_test, gt_test, correlations, num_selected, regressor):
    reg_scores = []
    reg_mses = []
    reg_maes = []
    most_correlated_indexes = list(np.argsort(correlations)[::-1][:num_selected])
    X = X[:, most_correlated_indexes]
    X_test = X_test[:, most_correlated_indexes]
    y = gt - np.mean(gt)
    y_test = gt_test - np.mean(gt)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)
    model = regressor.fit(X_scaled, y)
    reg_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    reg_mse = mse(y_test, y_pred)
    reg_mae = mae(y_test, y_pred)

    reg_scores.append(reg_score)
    reg_mses.append(reg_mse)
    reg_maes.append(reg_mae)
    
    reg_score = np.mean(reg_scores)
    reg_rmse = mse(y_test, y_pred, squared=False)
    reg_mae_std = np.std(np.abs(y_test-y_pred))
    reg_mae = np.mean(np.abs(y_test-y_pred))
    return regressor, reg_score, reg_rmse, reg_mae, reg_mae_std, np.mean(gt), reg_maes


train_data_df = pd.read_csv('../dataframes/DHCP_train_nn-UNet_extracted_features.csv', index_col=0)
test_data_df = pd.read_csv('../dataframes/DHCP_test_nn-UNet_extracted_features.csv', index_col=0)
selected_size = 100 # number of features selected among those with most correlations with age
regressor = BayesianRidge()


variable = 'scan_age' #it can be one of ['birth_age', 'scan_age', 'birth_weight', 'head_circumference']
gender = 'all' #it can be 'Male' or 'Female'
X_train, y_train = prepare_data(train_data_df, gender, variable)
X_test, y2_test = prepare_data(test_data_df, gender, variable)
correlations = pearson_correlation(X_train, y_train)
regressor, scaler, reg_score, reg_rmse, reg_mae, reg_mae_std, bias = full_regression(X_train, y_train, X_test, y2_test, regressor)
regressor, reg_score_sel, reg_rmse_sel, reg_mae_sel, reg_mae_std_sel, bias, maes = most_correlated_regression(X_train, y_train, X_test, y2_test,  correlations, selected_size, regressor)
print('** No selection **  MAE: {:.4f}±{:.4f},  R2: {:.4f},   RMSE: {:.4f}'.format(reg_mae, reg_mae_std, reg_score, reg_rmse))
print('** With selection **  MAE: {:.4f}±{:.4f},  R2: {:.4f},   RMSE: {:.4f}'.format(reg_mae_sel, reg_mae_std_sel, reg_score_sel, reg_rmse_sel))





