import MLtools
import visualisation
import dataprocessing
import pandas as pd
import ipywidgets as widgets
import numpy as np
from IPython.display import display, clear_output
import statistics

def createUI(weekdays: dict, hours: dict, trips: dict, heatmap: np.ndarray, df: pd.DataFrame, static_list: list) -> widgets.Tab:
    """
    Creates all the UI components and returns the main tab for displaying.
    !Function doesn't get the variables from main.main() function automatically!
    """

    #Dataframe section of the UI:
    #Creating the button that outputs the dataframe.
    def dataframeButtonPressed(button):
        """Clear the output of the output_df widget and display the dataframe in it."""
        with output_df:
            clear_output()
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                display(df)
    dataframe_button = widgets.Button(
        description='Load DataFrame',
        tooltip='Load DataFrame',
    )
    dataframe_button.on_click(dataframeButtonPressed)

    #Creating the actual tab that will show the dataframe.
    output_df = widgets.Output()
    df_vbox = widgets.VBox([output_df], layout=widgets.Layout(height='400px', overflow='auto', width='1200px', flex_flow='row', display='flex'))
    df_view = widgets.VBox([dataframe_button, df_vbox])
    machine_tabs = widgets.Tab()

    #Graph drawing section of the UI:
    #Creating the dropdown menus for selecting which visualisation to use.
    desc_drop = widgets.Dropdown(
        options=[''],
        value='',
        description=':',
        disabled=False,
    )
    visualization_drop = widgets.Dropdown(
        options=[''],
        value='',
        description=':',
        disabled=False,
    )

    #Creating the button that will render the graphs.
    def graphButtonPressed(button):
        """Clear the output of the trips_output widget and display the graphs in it."""
        with trips_output:
            clear_output()
            data = None
            if visualization_drop.value == "weekdays":
                data = weekdays
            elif visualization_drop.value == "hours":
                data = hours
            else:
                data = visualisation.makeDict(trips)
            a = visualisation.makeVisualization(desc_drop.value, visualization_drop.value, data, trips, hours)
            a.show()
    graph_button = widgets.Button(
        description='Visualize',
        tooltip='Visualize',
    )

    #Creating the actual tab that will render the graphs.
    trips_output = widgets.Output()
    graph_button.on_click(graphButtonPressed)
    trips_rend = widgets.VBox([trips_output])
    trips_vbox = widgets.VBox([desc_drop, visualization_drop, graph_button, trips_rend])

    #Heatmap section of the UI:
    #Button for drawing the heatmap.
    def heatmapButtonPressed(button):
        """Clear the output of the heatmap_output widget and display the heatmap in it."""
        with heatmap_output:
            clear_output()
            visualisation.getHeatMap(heatmap)
    heatmap_button = widgets.Button(
        description='Draw Heatmap',
        tooltip='Draw Heatmap',
    )
    heatmap_button.on_click(heatmapButtonPressed)

    #Actual frame where the heatmap will be rendered.
    heatmap_output = widgets.Output()
    heatmap_vbox = widgets.VBox([heatmap_output], layout=widgets.Layout(height='700px', width='700px'))
    heatmap_view = widgets.VBox([heatmap_button, heatmap_vbox])

    #Static Points section of the UI:
    #Button that will draw the static points map.
    def staticButtonPressed(button):
        """Clear the output of the static_output widget and display the static points map in it."""
        with static_output:
            clear_output()
            a = visualisation.drawStaticsNodes(static_list)
            display(a)
    static_button = widgets.Button(
        description='Draw Static Points',
        tooltip='Draw Static Points',
    )
    static_button.on_click(staticButtonPressed)

    #Tab for showing the rendered static points map.
    static_output = widgets.Output()
    static_vbox = widgets.VBox([static_output], layout=widgets.Layout(height='700px', width='700px'))
    static_view = widgets.VBox([static_button, static_vbox])

    #Creating the tabs for data visualisation section of the UI.
    dataTabs = widgets.Tab()
    dataTabs.children = [trips_vbox, heatmap_view, static_view, df_view]
    dataTabs.set_title(0, '')
    dataTabs.set_title(1, '')
    dataTabs.set_title(2, '')
    dataTabs.set_title(3, '')

    categories = dataprocessing.getGroceries().keys()
    categories = list(categories)

    output_machine_df = widgets.Output()
    df_machine_vbox = widgets.VBox([output_machine_df], layout=widgets.Layout(height='400px', overflow='auto', width='1200px', flex_flow='row', display='flex'))

    category_dropdown2 = widgets.Dropdown(
        options=categories,
        description='Y:',
        disabled=False,
    )
    hourselect2 = widgets.IntSlider(
        value=6,
        min=6,
        max=22,
        step=1,
        description='Hour:',
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )
    weekdayselect = widgets.IntSlider(
        value=0,
        min=0,
        max=6,
        step=1,
        description='Weekday:',
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )
    checkboxes = []
    for category in categories:
        box = widgets.Checkbox(
            value=False,
            description=category,
            disabled=False,
            indent=False
        )
        checkboxes.append(box)
    checkbox_vbox = widgets.VBox(checkboxes)

    def machinebutton2Pressed(button):
        ml_df = MLtools.createMLDataframe(df, hourselect2.value)
        new_df = pd.DataFrame(columns=ml_df.columns)
        new_df.loc[0] = np.zeros(len(new_df.columns))
        new_df[""] = statistics.mean(ml_df[""])
        new_df[""] = statistics.mean(ml_df[""])
        new_df[""] = statistics.mean(ml_df[""])
        new_df[""] = statistics.mean(ml_df[""])
        new_df[""+str(hourselect2.value)] = 1
        new_df[""+str(weekdayselect.value)] = 1
        for box in checkboxes:
            if category_dropdown2.value == box.description:
                box.value = False
            new_df[box.description] = box.value and 1 or 0
        chance_percent = MLtools.linearSVC(ml_df, new_df, category_dropdown2.value)
        with machine_output2:
            clear_output()
            print(str(round(chance_percent[0][1]*100))+" %")
        with output_machine_df:
            clear_output()
            display(new_df)

    machine_button2 = widgets.Button(
        description='',
        tooltip='',
    )
    machine_button2.on_click(machinebutton2Pressed)

    machine_output2 = widgets.Output()

    jee_vbox = widgets.VBox([hourselect2, weekdayselect, category_dropdown2, checkbox_vbox, machine_button2, machine_output2])

    #Creating the tabs for machine learning section of the UI.
    machine_tabs = widgets.Tab()
    machine_tabs.children = [jee_vbox, df_machine_vbox]
    machine_tabs.set_title(0, "")
    machine_tabs.set_title(1, "")

    #Creating the main tab and nesting all the tabs created above in it.
    mainTabs = widgets.Tab()
    mainTabs.children = [dataTabs, machine_tabs]
    mainTabs.set_title(0, '')
    mainTabs.set_title(1, '')


    #Display the whole thing.
    return mainTabs
