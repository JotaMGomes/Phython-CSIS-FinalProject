"""
Project:    CSIS4260
            Fall 2019
Instructor: Nikhil Bhardwaj
Student:    Jose Luiz Mattos Gomes 
            #300291877
File:       ProjectJLMG_CSIS4260.py

"""

## Import libraries
import pandas as pd

from ProjectJLMG_CSIS4260_Defs import *

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# main code #

## Importing data as Dataframe

### Import features.csv
df_features = pd.read_csv('./Data/features.csv', index_col=[0,1])

### Construct data for prediction (Dates > 2012-10-26)
df_predict_features = df_features.loc[df_features.index.get_level_values('Date') > '2012-10-26']

### Import train.csv
df_import_csv = pd.read_csv('./Data/train.csv', usecols = ['Store','Dept','Date','Weekly_Sales'])

#### Rename column
df_import_csv.rename(columns={'Weekly_Sales':'Sales'}, inplace=True)

#clean data and remove negative values
df_import_csv['Sales'] = df_import_csv['Sales'].apply(lambda x: 0 if x < 0 else x)

## Create store list
store_list = df_features.index.levels[0]

## Create department list
dept_list = df_import_csv['Dept'].unique()

# define resulting dataframes
p_corr_final = pd.DataFrame()

# define Root Mean Square error dataframe
df_RMS = pd.DataFrame(columns=['Store','Dept', 'Size', 'Model', 'MinErr'])
df_RMS.set_index(['Store', 'Dept'], inplace=True)

# final dataframe with prediction
df_final_predict = pd.DataFrame(columns=['Store','Dept', 'Date', 'FutureValue'])
df_final_predict.set_index(['Store', 'Dept', 'Date'], inplace=True)

# set inkex key list
key_list = ['Store', 'Dept']
idx = pd.IndexSlice

# max iterations for debugging
# Store: 45 | Dept: 65 | Count: 3645
# max_count = 1
# max_count = 20

# max iterations for production
max_count = 4000  

# define auxiliar counter
num_count = 1  

## iterate through Stores
for v_store in store_list:

    ## iterate through Depts
    for v_dept_aux in dept_list:

        # convert numpy.int64 to int
        v_dept = v_dept_aux.item()
       
        # print current calculations
        print()
        print('Store: ' + str(v_store) + ' | Dept: ' + str(v_dept) + ' | Count: ' + str(num_count))

        ## Create subset dataFrame from df_import_csv
        df_subset01 = df_import_csv.loc[(df_import_csv['Store']==v_store) & (df_import_csv['Dept']==v_dept)].set_index(['Date'])

        ## Create subset dataFrame from features
        df_subset02 = df_features.xs(v_store)

        ## Merge data
        df_calc = pd.merge(df_subset01, df_subset02, on='Date', how='left')

        # calculate only if data exists for this subset of store / Dept 
        if len(df_calc) > 0:

           ## Create subset dataFrame from df_predict_features
           df_subset03 = df_predict_features.loc[idx[v_store, :],[]]
           df_subset03['Dept'] = v_dept

           # set index
           df_subset03.set_index('Dept', append=True, inplace=True)
           df_subset03 = df_subset03.reorder_levels(['Store', 'Dept', 'Date'])
                     
           # Add number of entries for this subset
           df_RMS.loc[(v_store, v_dept),'Size'] = len(df_calc)

           # split data into training (80%) and test (20%)
           (train_set, test_set) = splitData(df_calc[['Sales']])

           ## reset min error variable value to a max value for comparison
           min_rms = 999999999

           ## do Simple Mean
           df_predict_temp = pd.DataFrame(columns=['Store','Dept', 'Date'])
           (df_RMS.loc[(v_store, v_dept),'SimpleMean'], df_predict_temp)  = doSimpleMean(train_set, test_set, df_subset03)

           # Test min error
           if df_RMS.loc[(v_store, v_dept),'SimpleMean'] < min_rms:
              min_rms = df_RMS.loc[(v_store, v_dept),'SimpleMean']
              df_RMS.loc[(v_store, v_dept),'Model']='SimpleMean'
              df_RMS.loc[(v_store, v_dept),'MinErr']= min_rms
              df_final_predict = updateFinalPrediction(df_final_predict, df_predict_temp)


           ## do Naive
           df_predict_temp = pd.DataFrame(columns=['Store','Dept', 'Date'])
           (df_RMS.loc[(v_store, v_dept),'Naive'], df_predict_temp)  = doNaive(train_set, test_set, df_subset03)
           
           # Test min error
           if df_RMS.loc[(v_store, v_dept),'Naive'] < min_rms:
              min_rms = df_RMS.loc[(v_store, v_dept),'Naive']
              df_RMS.loc[(v_store, v_dept),'Model']='Naive'
              df_RMS.loc[(v_store, v_dept),'MinErr']= min_rms
              df_final_predict = updateFinalPrediction(df_final_predict, df_predict_temp)
           

           ## do Moving Avg 5 -> # 1 month
           df_predict_temp = pd.DataFrame(columns=['Store','Dept', 'Date'])
           (df_RMS.loc[(v_store, v_dept),'MovingAvg5'], df_predict_temp) = doMovingAvgN(train_set, test_set, df_subset03, 5)    

           # Test min error
           if df_RMS.loc[(v_store, v_dept),'MovingAvg5'] < min_rms:
              min_rms = df_RMS.loc[(v_store, v_dept),'MovingAvg5']
              df_RMS.loc[(v_store, v_dept),'Model']='MovingAvg5'
              df_RMS.loc[(v_store, v_dept),'MinErr']= min_rms
              df_final_predict = updateFinalPrediction(df_final_predict, df_predict_temp)

           
           ## do Moving Avg 27 -> 6 month
           df_predict_temp = pd.DataFrame(columns=['Store','Dept', 'Date'])
           (df_RMS.loc[(v_store, v_dept),'MovingAvg27'], df_predict_temp) = doMovingAvgN(train_set, test_set, df_subset03, 27)    

           # Test min error
           if df_RMS.loc[(v_store, v_dept),'MovingAvg27'] < min_rms:
              min_rms = df_RMS.loc[(v_store, v_dept),'MovingAvg27']
              df_RMS.loc[(v_store, v_dept),'Model']='MovingAvg27'
              df_RMS.loc[(v_store, v_dept),'MinErr']= min_rms
              df_final_predict = updateFinalPrediction(df_final_predict, df_predict_temp)


           ## do Moving Avg 53 -> 12 month
           df_predict_temp = pd.DataFrame(columns=['Store','Dept', 'Date'])
           (df_RMS.loc[(v_store, v_dept),'MovingAvg53'], df_predict_temp) = doMovingAvgN(train_set, test_set, df_subset03, 53)    

           # Test min error
           if df_RMS.loc[(v_store, v_dept),'MovingAvg53'] < min_rms:
              min_rms = df_RMS.loc[(v_store, v_dept),'MovingAvg53']
              df_RMS.loc[(v_store, v_dept),'Model']='MovingAvg53'
              df_RMS.loc[(v_store, v_dept),'MinErr']= min_rms
              df_final_predict = updateFinalPrediction(df_final_predict, df_predict_temp) 


           ## do Holt Linear
           df_predict_temp = pd.DataFrame(columns=['Store','Dept', 'Date'])
           (df_RMS.loc[(v_store, v_dept),'HoltsLinear'], df_predict_temp) = doHoltsLinear(train_set, test_set, df_subset03)
           
           # Test min error
           if df_RMS.loc[(v_store, v_dept),'HoltsLinear'] < min_rms:
              min_rms = df_RMS.loc[(v_store, v_dept),'HoltsLinear']
              df_RMS.loc[(v_store, v_dept),'Model']='HoltsLinear'
              df_RMS.loc[(v_store, v_dept),'MinErr']= min_rms
              df_final_predict = updateFinalPrediction(df_final_predict, df_predict_temp)


           ## do Holt Winter
           df_predict_temp = pd.DataFrame(columns=['Store','Dept', 'Date'])
           (df_RMS.loc[(v_store, v_dept),'HoltWinter'], df_predict_temp) = doHoltWinter(train_set, test_set, df_subset03)

           # Test min error
           if df_RMS.loc[(v_store, v_dept),'HoltWinter'] < min_rms:
              min_rms = df_RMS.loc[(v_store, v_dept),'HoltWinter']
              df_RMS.loc[(v_store, v_dept),'Model']='HoltWinter'
              df_RMS.loc[(v_store, v_dept),'MinErr']= min_rms
              df_final_predict = updateFinalPrediction(df_final_predict, df_predict_temp)


           ## do ARIMA
           df_predict_temp = pd.DataFrame(columns=['Store','Dept', 'Date'])
           (df_RMS.loc[(v_store, v_dept),'Arima'], df_predict_temp) = doArima(train_set, test_set, df_subset03)

           # Test min error
           if df_RMS.loc[(v_store, v_dept),'Arima'] < min_rms:
              min_rms = df_RMS.loc[(v_store, v_dept),'Arima']
              df_RMS.loc[(v_store, v_dept),'Model']='Arima'
              df_RMS.loc[(v_store, v_dept),'MinErr']= min_rms
              df_final_predict = updateFinalPrediction(df_final_predict, df_predict_temp)

           
           ## Multi regression

           # ## calculate Pearson's correlation

           # create dataframe to stores correlation indexes
           pc = df_calc.corr(method ='pearson')['Sales']
     
           p_corr = pd.DataFrame({'Store':v_store, 'Dept':v_dept, 'Temperature':pc['Temperature'], 'Fuel_Price':pc['Fuel_Price'],
           'MarkDown1':pc['MarkDown1'],'MarkDown2':pc['MarkDown2'],'MarkDown3':pc['MarkDown3'],'MarkDown4':pc['MarkDown4'],
           'MarkDown5':pc['MarkDown5'],'CPI':pc['CPI'],'Unemployment':pc['Unemployment'],'IsHoliday':pc['IsHoliday']}, index =[num_count])
     
           p_corr_final = pd.concat([p_corr_final, p_corr])
     
           # remove the keys from df
           p_corr = p_corr.drop(columns=key_list)
           
           # create emply list of columns to drop
           column_list = []
     
           # Choose variables with correlation value greater then abs(0.3)
           for (columnName, columnData) in p_corr.iteritems():
               if not is_nan(columnData.values):
                  if abs(columnData.values) >= 0.3:
                     column_list.append(columnName)
     
           # only call algorithm if there is one or more independent variables
           if len(column_list) > 0:
     
              # Remove NaN values from df_calc
              df_calc.fillna(method='ffill',inplace=True)
              df_calc.fillna(method='bfill',inplace=True)

              ## Create subset dataFrame from df_predict_features
              df_subset03 = df_predict_features.loc[idx[v_store, :],column_list]
              df_subset03['Dept'] = v_dept
              df_subset03.set_index('Dept', append=True, inplace=True)
              df_subset03 = df_subset03.reorder_levels(['Store', 'Dept', 'Date'])

              # Remove NaN values from df_subset03
              df_subset03.fillna(method='ffill',inplace=True)
              df_subset03.fillna(method='bfill',inplace=True)
              
              # call Linear Regression
              (df_RMS.loc[(v_store, v_dept),'LinearRegression'], df_predict_temp) = doLinearRegression(df_calc[column_list], df_calc['Sales'], df_subset03, v_dept)

              # Test min error
              if df_RMS.loc[(v_store, v_dept),'LinearRegression'] < min_rms:
                 min_rms = df_RMS.loc[(v_store, v_dept),'LinearRegression']
                 df_RMS.loc[(v_store, v_dept),'Model']='LinearRegression'
                 df_RMS.loc[(v_store, v_dept),'MinErr']= min_rms
                 df_final_predict = updateFinalPrediction(df_final_predict, df_predict_temp)

           else:

              # as the algorithm was not called, set max error
              df_RMS.loc[(v_store, v_dept),'LinearRegression'] = 999999999

        # update counter
        num_count = num_count + 1

        # if current couter is grater than max_counter, exit loop
        if num_count > max_count:
           break

    # if current couter is grater than max_counter, exit loop
    if num_count > max_count:
       break


# create index to p_corr_final df
p_corr_final.set_index(['Store','Dept'], inplace=True)

# Export data
p_corr_final.to_csv('./Export/p_corr_final.csv')
df_RMS.to_csv('./Export/RMS.csv')
df_final_predict.to_csv('./Export/Predict_Final.csv')

