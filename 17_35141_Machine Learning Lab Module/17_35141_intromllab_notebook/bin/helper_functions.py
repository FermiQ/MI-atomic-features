# This .py file contains what we're calling "helper" functions used throughout the lab activities. these are put seperately to clean up the look and feel of the lab.

import pandas as pd              # Pandas is a data analysis library which we'll primarily use to handle our dataset
import numpy as np               # Numpy is a package for scientific computing. We'll use it for some of it's math functions
import matplotlib.pyplot as plt
from sklearn import metrics

def parity_plot(y_true,Y_predictions,title="Parity Plot"):
    fig1,ax1 = plt.subplots()
    ax1.scatter(y_true, Y_predictions, edgecolors=(0, 0, 0))
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=4)
    ax1.legend(["Ideal", "Prediction"])
    ax1.set_xlabel('Measured Bandgap (eV)')
    ax1.set_ylabel('Predicted Bandgap (eV)')
    ax1.set_title(title)
    plt.show()

def parity_plots_side_by_side(y_left_true,Y_left_predictions,y_right_true,Y_right_predictions,title_left="Training Parity Plot",title_right="Test Parity Plot"):
    fig1, (ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
    ax1.scatter(y_left_true, Y_left_predictions, edgecolors=(0, 0, 0))
    ax1.plot([y_left_true.min(), y_left_true.max()], [y_left_true.min(), y_left_true.max()], 'k--', lw=4)
    ax1.legend(["Ideal", "Prediction"])
    ax1.set_xlabel('Measured Bandgap (eV)')
    ax1.set_ylabel('Predicted Bandgap (eV)')
    ax1.set_title(title_left)
    
    ax2.scatter(y_right_true, Y_right_predictions, edgecolors=(0, 0, 0))
    ax2.plot([y_right_true.min(), y_right_true.max()], [y_right_true.min(), y_right_true.max()], 'k--', lw=4)
    ax2.legend(["Ideal", "Prediction"])
    ax2.set_xlabel('Measured Bandgap (eV)')
    ax2.set_ylabel('Predicted Bandgap (eV)')
    ax2.set_title(title_right)
    
    plt.show()

def rmse(y_true, Y_predictions): return round(np.sqrt(metrics.mean_squared_error(y_true, Y_predictions)),4)
def rmse_std(y_true, Y_predictions): return round(rmse(y_true, Y_predictions)/np.std(y_true),4)
def mae(y_true, Y_predictions): return round(metrics.mean_absolute_error(y_true, Y_predictions),4)
def r2(y_true, Y_predictions): return round(metrics.r2_score(y_true, Y_predictions),4)

def parity_stats(y_true,Y_predictions):
    
    print("RMSE: ", rmse(y_true,Y_predictions), "(0.0 for perfect prediction)")
    print("RMSE/std: ", rmse_std(y_true,Y_predictions), "(0.0 for perfect prediction)")
    print("MAE: ", mae(y_true,Y_predictions), "(0.0 for perfect prediction)")
    print("R2: ", r2(y_true,Y_predictions), "(1.0 for perfect prediction)")
    return rmse(y_true,Y_predictions),rmse_std(y_true,Y_predictions),mae(y_true,Y_predictions),r2(y_true,Y_predictions)

def parity_stats_side_by_side(y_left_true,Y_left_predictions,y_right_true,Y_right_predictions,title_left,title_right):
    
    rmse_left = rmse(y_left_true,Y_left_predictions)
    rmse_std_left = rmse_std(y_left_true,Y_left_predictions)
    mae_left = mae(y_left_true,Y_left_predictions)
    r2_left = r2(y_left_true,Y_left_predictions)
    
    rmse_right = rmse(y_right_true,Y_right_predictions)
    rmse_std_right = rmse_std(y_right_true,Y_right_predictions)
    mae_right = mae(y_right_true,Y_right_predictions)
    r2_right = r2(y_right_true,Y_right_predictions)
    
    stats_df = pd.DataFrame({'Error Metric' : ['RMSE', 'RMSE/std', 'MAE', 'R2'], 
                             title_left : [str(rmse_left) + " (eV)", rmse_std_left, str(mae_left) + " (eV)", r2_left],
                             title_right: [str(rmse_right) + " (eV)", rmse_std_right, str(mae_right) + " (eV)", r2_right],
                             'Note' : ['(0.0 for perfect prediction)', '(0.0 for perfect prediction)','(0.0 for perfect prediction)','(1.0 for perfect prediction)']})
    return stats_df

# print the average performance of models with the best hyperparameter during grid search
def CV_best_stats(CV,y_true):
    idx = CV.best_index_
    
    rmse = round(np.sqrt(-CV.cv_results_['mean_test_neg_mean_squared_error'][idx]),4)
    rmse_std = round(rmse/np.std(y_true),4)
    mae = round(-CV.cv_results_['mean_test_neg_mean_absolute_error'][idx],4)
    r2 = round(CV.cv_results_['mean_test_r2'][idx],4)
    
    print("Average test RMSE: ", rmse, "(0.0 for perfect prediction)")
    print("Average test RMSE/std: ", rmse_std, "(0.0 for perfect prediction)")
    print("Average test MAE: ", mae, "(0.0 for perfect prediction)")
    print("Average test R2: ", r2, "(1.0 for perfect prediction)")
    return rmse,rmse_std,mae,r2

# average bandgaps where we have multiple sources for the same composition and drop duplicate compositions
def average_bandgaps(master_df, input_col_header, output_col_header):
#     diff_list = list()
    for chem_formula in master_df[input_col_header].unique():
        temp_df = master_df.copy()
        temp_df = temp_df.loc[temp_df[input_col_header]==chem_formula]
        if len(temp_df) > 1:
            avg_bandgap = temp_df[output_col_header].mean()
            #diff = temp_df[output_col_header].max()-temp_df[output_col_header].min()
            #diff_list.append(diff)
            indexes = temp_df.index
            updated_master_df = master_df.copy()
            updated_master_df.at[indexes,output_col_header] = avg_bandgap
    master_df_clean = updated_master_df.drop_duplicates(subset=input_col_header)
    return master_df_clean