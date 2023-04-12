import dataprocessing as dp
import MLtools as ml
import system
import ui
import visualisation as vis
import ui
from IPython.display import clear_output
import os, sys

def main():
    system.installation()
    clear_output()

    print('')
    df = dp.getData()
    clear_output()

    print('')
    df = dp.cleanData(df)
    static_list, moving_list = dp.getStaticsNode(df)
    clear_output()

    print('')
    sorted_trips = dp.separateTrips(moving_list)
    clear_output()

    print('')
    weekdays, hours, cleaned_trip, heatmap = dp.tripCleaner(sorted_trips)
    df = dp.createDataFrame(cleaned_trip)
    clear_output()

    return ui.createUI(weekdays, hours, cleaned_trip, heatmap, df, static_list)
    #return weekdays, hours, cleaned_trip, heatmap, df, static_list
