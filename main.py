import reverse_geocode as revgc
import plotly.express as px
import holidays
import pandas as pd
import main as main
import numpy as np
import holidays
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
#from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error

def FindingCity(row,point):
    if point.lower() == 'start':
        coordinates = [(row['start station latitude'],row['start station longitude'])]
    else:
        coordinates  = [(row['end station latitude'],row['end station longitude'])]
    
    return revgc.search(coordinates)[0]['city']

def DatetimeInterval(df,freq):
    
    df1 = df.set_index('starttime')
    try:
        df1 = pd.get_dummies(df1, columns=['usertype','cluster_label'])
        df1 = df1.resample(rule=freq, label='left', origin='start_day').sum()
        pickups = df1.iloc[:,-7:]             
    except KeyError:
        #print(KeyError)
        df1 = pd.get_dummies(df1, columns=['usertype'])
        df1 = df1.resample(rule=freq, label='left', origin='start_day').sum()
        pickups = df1.iloc[:,-2:]        
    
    pickups['pickups'] = pickups.loc[:,['usertype_Customer','usertype_Subscriber']].sum(axis=1)
    cluster_pickups = pd.DataFrame(pickups)
    
    return cluster_pickups
    #pickups = pickups['tripduration'].groupby(pd.Grouper(freq=freq, label='left', origin='start_day')).count()    
    #pickups.rename(columns={'tripduration':'pickups'}, inplace=True)


def PreciptypeMap(row):
    if row == 'rain':
        return 2
    elif row == 'rain,snow':
        return 3
    elif row == 'snow':
        return 4
    elif row in ['freezingrain','snow,ice']:
        return 5
    else:
        return 1

def ConditionsMap(row):
    if row in ['Overcast','Partially cloudy','Clear']:
        return 1
    elif row in ['Rain, Overcast','Rain, Partially cloudy','Snow, Partially cloudy']:
        return 2
    elif row in ['Snow, Rain, Overcast','Snow, Overcast','Snow, Ice, Overcast']:
        return 3

def MergingDataFrames(wd, df):
    pickups = pd.merge(wd, df, left_index=True, right_index=True)
    return pickups

def DataPreprocess(pickup,season_dict):
    pickup['workingday'] = pickup.index.map(lambda row: 1 if row.dayofweek < 5 else 0)
    pickup['holiday'] = pickup.index.map(lambda row: 1 if row in holidays.US() else 0)
    pickup['month'] = pickup.index.month
    pickup['hour'] = pickup.index.hour
    pickup['minute'] = pickup.index.minute
    pickup.reset_index(drop=True, inplace=True)
    pickup['season'] = pickup['month'].map(season_dict)
    
    return pickup


def Lagging(df, minutes):
    
    df['month'] = df.index.month
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute

    for i in range(int(((120/minutes)/2) +1), int((120/minutes) +1)):
        df['lag(pickups,{}-{})'.format(i*minutes-minutes,i*minutes)] = df['pickups'].shift(i)
    df = df.dropna().drop(['usertype_Customer','usertype_Subscriber'], axis=1)
    return df


def WeatherLaging(df, minutes, weather_columns):

    df.loc[:,weather_columns] = df.loc[:,weather_columns].shift(int(120/minutes/2))

    for i in range(int(((120/minutes)/2) +1), int((120/minutes) +1)):
        df['lag(pickups,{}-{})'.format(i*minutes-minutes,i*minutes)] = df['pickups'].shift(i)
    
    df = df.dropna(axis=0, how='any')

    return df

def get_location_interactive(df, mapbox_style="open-street-map"):
    """Return a map with markers for houses based on lat and long.
    
    Parameters:
    ===========
    mapbox_style = str; options are following:
        > "white-bg" yields an empty white canvas which results in no external HTTP requests
        > "carto-positron", "carto-darkmatter", "stamen-terrain",
          "stamen-toner" or "stamen-watercolor" yield maps composed of raster tiles 
          from various public tile servers which do not require signups or access tokens
        > "open-street-map" does work 'latitude', 'longitude'
    """
    fig = px.scatter_mapbox(
        df,
        lat=df['start station latitude'],
        lon=df['start station longitude'],
        size = 'cluster_labelcount',
        color='cluster_label',
        #color_continuous_scale=["green", 'blue', 'red', 'gold',''],
        zoom=11.5,
        range_color=[0, df['cluster_label'].quantile(0.95)], # to negate outliers
        height=700,
        title='Station location',
        opacity=.5,
        center={
            'lat': df['start station latitude'].mode()[0],
            'lon': df['start station longitude'].mode()[0]
        })
    fig.update_layout(mapbox_style=mapbox_style)
    #fig.update_layout(margin={"r": 0, "l": 0, "b": 0})
    fig.show()
    pass

class PredictionPipeline():
    
    def __init__(self, pickups, weather_columns):
        self.pickups = pickups
        self.weather_columns = weather_columns
        
    def PredictionDataPreperation(self):
    
        X = self.pickups.drop(['pickups'], axis=1)
        y = self.pickups[['month','hour','pickups']]
    
        scaler = StandardScaler()
        lag_columns = [c for c in X.columns if c.startswith("lag")]

        if len(self.weather_columns)>0:
            X[self.weather_columns+lag_columns] = scaler.fit_transform(X[self.weather_columns+lag_columns])
        else:
            X[lag_columns] = scaler.fit_transform(X[lag_columns])
    
        X_train = X[X['month']!=8]
        X_test = X[X['month']==8]
        y_train = y[y['month']!=8]
        y_test = y[y['month']==8]    
        
        return X_train, X_test, y_train, y_test
    
    def BackTestingPrediction(self, model, model_name, X_train, X_test, y_train, y_test):
    
        print('\n',model_name)
        CV = []
        CV_score = []
        CV_MSE = []
        for m in  X_train.month.unique()[:-1]:
            CV.append(m)
            xtemp_train = X_train[X_train['month'].isin(CV)]
            ytemp_train = y_train[y_train['month'].isin(CV)]['pickups']
            xtemp_test = X_train[X_train['month']==m+1]
            ytemp_test = y_train[y_train['month']==m+1]['pickups']
    
            model.fit(xtemp_train, ytemp_train)
            train_temp_preds = model.predict(xtemp_train)
            test_temp_preds = model.predict(xtemp_test)
            print('\n Train Score: ',  model.score(xtemp_train, ytemp_train))
            print(' Train RMSE : ',  np.sqrt(mean_squared_error(ytemp_train, train_temp_preds)))
            print(' Test Score:    ',  model.score(xtemp_test, ytemp_test))
            print(' Test RMSE :    ',  np.sqrt(mean_squared_error(ytemp_test, test_temp_preds)))
        
            CV_score.append(model.score(xtemp_test, ytemp_test))
            CV_MSE.append(np.sqrt(mean_squared_error(ytemp_test, test_temp_preds)))
        
    
        model.fit(X_train, y_train['pickups'])          
    
        #print('\nTrain Score: ',  model.score(X_train, y_train['pickups']))
        #print('Train RMSE : ',  np.sqrt(mean_squared_error(y_train['pickups'], model.predict(X_train))))
        #print('Test Score: ', model.score(X_test, y_test['pickups']))
        #print('Test RMSE :   ', np.sqrt(mean_squared_error(y_test['pickups'], model.predict(X_test))))

        try:
            fimpot = [(feature,importance) for (feature,importance) in zip(X_train.columns,model.feature_importances_)]
            #for feature, impostance in zip(X_train.columns,model.feature_importances_):
                #print('\n{} : {:0.4f}'.format(feature,impostance))
        except:
            fimpot = [(feature,importance) for (feature,importance) in zip(X_train.columns,model.coef_)]
            #for feature, impostance in zip(X_train.columns,model.coef_):
                #print('\n{} : {:0.4f}'.format(feature,impostance))


        return CV_score, CV_MSE, model.score(X_test, y_test['pickups']), np.sqrt(mean_squared_error(y_test['pickups'], model.predict(X_test))), fimpot, model.predict(X_test)