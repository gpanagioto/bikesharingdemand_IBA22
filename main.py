import reverse_geocode as revgc
import plotly.express as px
 
def FindingCity(row,point):
    if point.lower() == 'start':
        coordinates = [(row['start station latitude'],row['start station longitude'])]
    else:
        coordinates  = [(row['end station latitude'],row['end station longitude'])]
    
    return revgc.search(coordinates)[0]['city']

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