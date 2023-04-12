import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import ipywidgets as widgets
import numpy as np
import dataprocessing as dp
import seaborn as sns
import ipywidgets as widgets
import pandas as pd
import matplotlib.colors as mcolors

PICTURE = './Documents/cropped.jpg'

def getHeatMap(trips: list) -> plt.show:
    """How this work is: how many numbers are in list, that intense the heatmap is. 
    If list contains 2 number 19 (for example list: [18,19,19,20])
    it means that in grid cell 19(2nd row from bottom and 9th column from left) there is number 2"""
   
    # Fit filled gridZ**ero to gridList and reshape that it fits to grid
    gridList = trips
    side_len = int(dp.getGridSize()**0.5)
    gridList = np.reshape(gridList, (side_len, side_len))
    gridList = np.flip(gridList, 0)

    # Load shop layout
    img = Image.open(PICTURE)
    
    width, height = img.size
    print(img.size)

    # Make the heatmap by ploting shoplayout and grid
    fig, ax = plt.subplots(figsize=(12,12))
    am = sns.heatmap(gridList, cmap='Reds', linewidth=1, alpha = 0.7, cbar=False, xticklabels=False, yticklabels=False, ax=ax)
    am.imshow(img,
          aspect = am.get_aspect(),
          extent = am.get_xlim() + am.get_ylim(),
          zorder = 0)

    return plt.show()


def makeDict(trips: dict) -> dict:
    """Function takes dictionary and return a dictionary of values and keys from the first dictionary."""
    data1 = {}
    for x in trips.keys():

        data1[x] = list_1
 
    return data1

def makeVisualization(descrip: str, visualization: str, data: dict, trips: dict, hours: dict) -> plt:
    """At this function, fucntion check after we select something from ipywidgets, what visualization it is, and checking inside that descrip
    what we looking, it can be like this, "weekdays, avarage duration" or something else what u select. It's return end of every if visualization 
    list of what we select to descrip. """
    if visualization == 'weekdays':
        y_list = []
        x_list = ['Mon' , 'Tue'  , 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        xlabel = 'Day'
        palet = 'flare'  
       
        if descrip == '':
            ylabel = ''
            title = '' 
            statistics = ''
                         
        if descrip == '':
            ylabel = ''
            title = ''
            statistics = ''
 
        if descrip == '':
            ylabel = ''
            title = ''
            statistics = ''
            
        for day in list(data.keys()):
            y_list.append(data[day][statistics])     
        
    if visualization == 'nodes':   
        y_list = []
        x_list = list(trips)
        xlabel = 'Node'
        palet = 'husl'
        
        if descrip == '':
            ylabel = ''
            title = ''  
            statistics = ''
            
        if descrip == '':
            ylabel = ''
            title = '' 
            statistics = ''
            
        if descrip == '':
            ylabel = ''
            title = '' 
            statistics = ''
            
        for x in list(data.keys()):
            y_list.append(data[x][statistics])
 
    if visualization == 'hours':
        y_list = [] 
        x_list = list(hours)
        xlabel = 'Hours' 
        palet = 'dark:#5A9_r'
        
        if descrip == '':
            ylabel = '' 
            title = ''
            statistics = ''
            
        if descrip == '':
            ylabel = '' 
            title = ''
            statistics = ''
            
        if descrip == '':
            ylabel = '' 
            title = ''
            statistics = ''
            
        for day in list(data.keys()):
            y_list.append(data[day][statistics])
 
    # visualization 
    fig, ax = plt.subplots(figsize=(12,6))
    ax = sns.barplot(x = x_list, y = y_list, palette=palet)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return plt


def drawStaticsNodes(df: list) -> Image:
    """Draws statics nodes to the map using the mean coordinate values. Takes a list as an argument that holds the statics nodes dataframes and returns Image."""
    
    # Get image size with this method
    img = Image.open(PICTURE)
    width, height = img.size

    # Create image widget
    file = open(PICTURE, "rb")
    image = file.read()
    widgets.Image(
        value=image,
        format='jpg',
        width=width,
        height=height,
    )

    # Draw on image
    d = ImageDraw.Draw(img)

    # Calibration of coordinates
    x_offset = 205  # x offset
    y_offset = -55 # offset
    x_scale = 1018 / 3500
    y_scale = 823 / 3000

    def scale_coords(x,y):
        xr = (x + x_offset) * x_scale
        yr = ((y + y_offset) * y_scale - (3000 * y_scale)) * -1
        return xr, yr

    new_df = pd.DataFrame(columns = ['node_id','x', 'y'])
    for node_df in df:
        x_mean = node_df['x'].mean().astype(int)
        y_mean = node_df['y'].mean().astype(int)
        node_id = node_df['node_id'].unique()[0]
        row = pd.DataFrame([[node_id, x_mean, y_mean]], columns = ['node_id','x', 'y'])
        new_df = pd.concat([new_df, row])

    new_df = new_df.reset_index()
    new_df = new_df.drop(['index'], axis = 1)
    font_path = '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf'
    font = ImageFont.truetype(font_path, 40) 
    colors = mcolors.CSS4_COLORS
    color_list = list(colors.keys())
    for index, row in new_df.iterrows():
        ind = (len(color_list) // (new_df.shape[0] * 2)) * index + 2
        (x,y) = scale_coords(row['x'], row['y'])
        d.text((10, 10 + 35 * index), f"{row['node_id']}", fill = colors[color_list[ind]], font = font)
        d.rectangle((x, y, x + 20, y + 20), fill = colors[color_list[ind]])
    
    return img
