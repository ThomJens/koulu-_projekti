import os, sys
path = f'{os.getcwd()}/Tools'
sys.path.append(path)

import dataprocessing as dp
import MLtools as ml
import system
import ui
import visualisation as vis
import ui
from IPython.display import clear_output



def main():
    system.installation()
    clear_output()
    
    print('Fetching data.')
    df = dp.getData()
    clear_output()
    
    print('Cleaning the data.')
    df = dp.cleanData(df)
    static_list, moving_list = dp.getStaticsNode(df)
    clear_output()
    
    print('Identifying shopping trips.')
    sorted_trips = dp.separateTrips(moving_list)
    clear_output()
    
    print('Polishing the results.')
    weekdays, hours, cleaned_trip, heatmap = dp.tripCleaner(sorted_trips)
    df = dp.createDataFrame(cleaned_trip)
    clear_output()
    
    return ui.createUI(weekdays, hours, cleaned_trip, heatmap, df, static_list)
