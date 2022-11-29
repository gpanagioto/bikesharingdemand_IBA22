import reverse_geocode as revgc
import plotly.express as px
import pandas as pd
import holidays

def FindingCity(row,point):
    if point.lower() == 'start':
        coordinates = [(row['start station latitude'],row['start station longitude'])]
    else:
        coordinates  = [(row['end station latitude'],row['end station longitude'])]
    
    return revgc.search(coordinates)[0]['city']

def DatetimeInterval(df,freq):
    df1 = df.set_index('starttime')
    df1 = pd.get_dummies(df1, columns=['usertype'])
    df1 = df1.resample(rule=freq, label='left', origin='start_day').sum()
    pickups = df1.loc[:,['usertype_Customer','usertype_Subscriber']]
    pickups['pickups'] = pickups.loc[:,['usertype_Customer','usertype_Subscriber']].sum(axis=1)
    #pickups = pickups['tripduration'].groupby(pd.Grouper(freq=freq, label='left', origin='start_day')).count()    
    pickups = pd.DataFrame(pickups)
    #pickups.rename(columns={'tripduration':'pickups'}, inplace=True)
    
    return pickups

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
        return 0

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
    for i,v in enumerate(range(minutes,120+1,minutes)):
        df['lag(pickups,{})'.format(i+1)] = df['pickups'].shift(i+1)
        
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
        color_continuous_scale=["green", 'blue', 'red', 'gold'],
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
    fig.update_layout(margin={"r": 0, "l": 0, "b": 0})
    fig.show()
    pass