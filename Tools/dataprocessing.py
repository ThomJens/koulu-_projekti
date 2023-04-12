import pandas as pd
import scipy
import numpy as np
from statistics import mode
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from collections import namedtuple
from multiprocessing import Pool
from copy import deepcopy

GROCERYDICT = {

}

NODE_ID = {}

GRID_MIN = 0
GRID_MAX = 0
X_MIN = 0
X_MAX = 0
Y_MIN = 0
Y_MAX = 0

ADDRESS = 'postgresql+psycopg2://daika:d4ik4Duu@10.50.91.99:5432/iiwari_org'


def getGridSize():
    """Returns the grid size using formula: (grid_max - grid_min)**2"""
    return (GRID_MAX - GRID_MIN)**2


def getGroceries() -> list:
    """Returns the dictionary of groceries for ease of use."""
    return GROCERYDICT


def getNodeData(data_sql: str, replace: bool = True) -> pd.DataFrame:
    """Makes SQLquery and returns the result as dataframe. Takes SQLquery string as an argument."""
    alchemyEngine = create_engine(ADDRESS, pool_recycle=3600);
    postgreSQLConnection = alchemyEngine.connect();
    df = pd.read_sql(data_sql, postgreSQLConnection)
    if replace:
        old_value = df['node_id'].loc[0]
        new_value = NODE_ID[old_value]
        df['node_id'] = df['node_id'].replace([old_value], new_value)
    return df
    
    
def getData() -> pd.DataFrame:
    """Data is pulled from the postgres database. Data includes: 'node_id, timestamp, x, y, z, q'."""

    node_ids = getNodeData('', False)
    unique_node_ids = node_ids[''].unique()
    sql_queries = []
    
    for index, node_id in enumerate(unique_node_ids):
        node_id_sql = f'SELECT * FROM - WHERE node_id = \'{node_id}\' ORDER BY -'
        sql_queries.append(node_id_sql)
        NODE_ID[node_id] = index

    with Pool() as p:
        results = p.map(getNodeData, sql_queries)
    
    df = pd.concat(results)
    return df


def cleanData(df: pd.DataFrame) -> pd.DataFrame:
    """Data is cleaned by removing points outside of the area, fixing timestamp, fixing node_id, and dropping unnecessary columns."""

    # Dropping unused features.
    df = df.drop([''], axis = 1)
    
    # Select the points inside the shop.
    df = df[df['x'].between(0, 0)]
    df = df[df['y'].between(0, 0)]
    
    # Fixing the timezone and removing all values below seconds.
    df['timestamp'] = df['timestamp'].astype('datetime64[s]')
    df.timestamp = df.timestamp.dt.tz_localize('UTC')
    df.timestamp = df.timestamp.dt.tz_convert('Europe/Helsinki')
    df = df.reset_index(drop = True)

    return df


def getStaticsNode(df: pd.DataFrame) -> list:
    """Separate the statistic nodesfrom the rest, by checking how many points is in the shopping area related to entrance/checkout area between 0 am and 5 am. Takes a dataframe as an argument and returns two lists. First has the statistic nodes and the second one has the rest."""
    statistic_nodes = []
    moving_nodes = []
    filtered = df[df['timestamp'].dt.hour.between(0, 4)]
    
    for node in filtered['node_id'].unique():
        nodes = filtered[filtered['node_id'] == node]
        node_shape = nodes.shape[0]
        y_list = nodes[nodes['y'] >= 0].shape[0]
        
        # Calculate the precentage of the points in the shopping area.
        fraction = y_list / node_shape
        if fraction > 0.99:
            statistic_nodes.append(df[df['node_id'] == node])
        else:
            moving_nodes.append(df[df['node_id'] == node])

    return statistic_nodes, moving_nodes  


def checkOutsiders(series: namedtuple) -> bool:
    """Check if given namedtuple is outside of areas of interest. Returns true if not inside the areas and false if inside the areas."""
    if checkEntrance(series) or checkCheckout(series) or checkShopArea(series):
        return False
    return True


def checkEntrance(series: namedtuple) -> bool:
    """Check if given namedtuple is within entrance area. Returns true if is within area and false if not."""
    entrance_x = (0, 0)
    entrance_y = (0, 0)
    if getattr(series, 'x') >= entrance_x[0] and  getattr(series, 'x') <= entrance_x[1] and getattr(series, 'y') >= entrance_y[0] and  getattr(series, 'y') <= entrance_y[1]:
        return True
    return False


def checkCheckout(series: namedtuple) -> bool:
    """Check if given namedtuple is within checkout area. Returns true if is within area and false if not."""
    checkout_x = (0, 0)
    checkout_y = (0, 0)
    if getattr(series, 'x') >= checkout_x[0] and  getattr(series, 'x') <= checkout_x[1] and getattr(series, 'y') >= checkout_y[0] and  getattr(series, 'y') <= checkout_y[1]:
        return True
    return False


def checkShopArea(series: namedtuple) -> bool:
    """Check if given namedtuple is within shop area. Returns true if is within area and false if not."""
    shop_x = (0, 0)
    shop_y = (0, 0)
    if getattr(series, 'x') >= shop_x[0] and  getattr(series, 'x') <= shop_x[1] and getattr(series, 'y') >= shop_y[0] and  getattr(series, 'y') <= shop_y[1]:
        return True
    return False


def getGridIDs(df: pd.DataFrame) -> list:
    """Converts x and y values from a DataFrame to a list of grid IDs. Takes a Pandas DataFrame as an argument and returns a list."""
    cell_x =  (X_MAX - X_MIN) / GRID_MAX
    cell_y =  (Y_MAX - Y_MIN) / GRID_MAX
    df['grid_id'] = (GRID_MAX * ((df['y'] - Y_MIN) // cell_y)) + ((df['x'] - X_MIN) // cell_x).astype(int)
    return df['grid_id'].values


def identifyGroceries(gList: list) -> list:
    """Takes a list of gridIDs and checks which products are on the grid. Uses the GROCERYDICT dictionary. Returns a list of products in string format."""
    returnList = []
    glist = set(gList)
    for category in GROCERYDICT:
        if any(gridNum in GROCERYDICT[category] for gridNum in glist):
            returnList.append(category)
    return returnList


def createDataFrame(trips: dict) -> pd.DataFrame:
    """Creates a dataframe for machinelearning model. Takes a dictionary as an argument, which holds all the data separated by node_id and then by trip_id."""
    trip_df = pd.DataFrame()
    temp = deepcopy(trips)
    for node in temp.keys():
        temp[node].pop('', None)
        temp[node].pop('', None)
        temp[node].pop('', None)
        temp[node].pop('', None)
        temp_df = pd.DataFrame(temp[node]).T
        trip_df = pd.concat([temp_df, trip_df])
        
    mlb = MultiLabelBinarizer()
    mlb.fit(trip_df[''])
    new_col_names = ["%s" % c for c in mlb.classes_]

    mlb_groceries = pd.DataFrame(mlb.fit_transform(trip_df['']), columns = new_col_names, index = trip_df.index)
    trip_df = pd.concat([trip_df, mlb_groceries], axis = 1)

    trip_df = trip_df.drop([''], axis=1)
    trip_df = trip_df.reset_index()
    trip_df = trip_df.drop([''], axis=1)
    
    return trip_df


def timeCaclulator(point1: namedtuple, point2: namedtuple) -> int:
    """Return time difference between two points. Takes two namedtuples as an arguments."""
    return pd.Timedelta(point2 - point1).seconds


def dinstanceCaclulator(point1: namedtuple, point2: namedtuple) -> int:
    """Calculate distance between two points with Pythagoras theorem. Takes two namedtuples as an arguments."""
    a = (getattr(point1, '') - getattr(point2, '')) **2
    b = (getattr(point1, '') - getattr(point2, '')) **2
    c = (a + b)**0.5 
    return int(c/100)


def tripSeparator(node_df: pd.DataFrame) -> pd.DataFrame:
    """Identify and separate trips from dataframe to dictionary. Takes a dataframe as an arugments and returns a list containing three dictionaries: weekdays, hours, and trips."""
    start_point = None
    end_point = None
    trip_id = 0
    node_id = node_df['node_id'].unique()[0]

    # Aggregating the data.
    node_df['index'] = node_df.index
    node = node_df.groupby([node_df.index // 5], as_index = False).mean().astype(int)
    node = node.drop(['node_id'], axis = 1)
    node['index'] = node.index

    row_tracker = 0
    trips = {}
    trips[node_id] = {}
    distance = 0
    indices = []
    avg_duration = []
    avg_length = []
    heatmap = np.zeros(getGridSize())

    weekdays_node = {}
    for day in range(7):
        weekdays_node[day] =  {}

    hours_node = {}
    for hour in range(24):
        hours_node[hour] =  {}

    # Iterating through every row/point. Check if the point is in the entrance, checkout or in the shopping area. Calculate the relevant statistics.
    for row in node.itertuples():

        # Check if the point is in the shopping area, entrance or in the checkout.
        if not checkOutsiders(row):

            # First point is skipped, because previous point is needed for our function to work as designed.
            if row_tracker == 0:
                previous = row
                row_tracker += 1
                continue

            # Check if the point is inthe shopping area and add the point id to the list and calculate the distance between current and previous point.
            if checkShopArea(row) and start_point is not None:
                indices.append(getattr(row, ''))
                distance += dinstanceCaclulator(row, previous)

            # Check if the previous point is in the entrance and current point in the shopping area. If true, then the shopping trip starts here.
            if checkEntrance(previous) and checkShopArea(row):
                start_point = node_df.iloc[getattr(row, '')]

            # Check if the current point is in the checkout and the previous point in the shopping area. If true, shopping trip ends here.
            if checkCheckout(row) and checkShopArea(previous) and start_point is not None:
                end_point = node_df.iloc[getattr(row, '')]
                distance += dinstanceCaclulator(row, previous)
                indices.insert(0, start_point[''])
                indices.append(end_point[''])
                indices = indices[1:]

                duration = timeCaclulator(start_point[''], end_point[''])
                day = getattr(start_point, '').weekday()
                hour = getattr(start_point, '').hour
                velocity = round(distance / duration, 2)
                
                if velocity <= 0 and duration <= 0:
                    
                    trip_df = node[node.index.isin(indices)]
                    grid_id = getGridIDs(trip_df)

                    trips[node_id][trip_id] = {}
                    trips[node_id][trip_id][''] = grid_id
                    trips[node_id][trip_id][''] = identifyGroceries(grid_id)
                    trips[node_id][trip_id][''] = duration
                    trips[node_id][trip_id][''] = distance
                    trips[node_id][trip_id][''] = velocity
                    trips[node_id][trip_id][''] = day
                    trips[node_id][trip_id][''] = hour
                    trips[node_id][trip_id][''] = len(indices)
                    trip_id += 1

                    weekdays_node[day][''] += duration
                    weekdays_node[day][''] += distance
                    weekdays_node[day][''] += 1

                    hours_node[hour][''] += duration
                    hours_node[hour][''] += distance
                    hours_node[hour][''] += 1

                    np.add.at(heatmap, grid_id.astype(int), 1)

                    avg_duration.append(duration)
                    avg_length.append(distance)

                start_point = None
                end_point = None
                row_tracker = 0
                distance = 0
                indices = []

            previous = row
            row_tracker += 1

    # Check if trips is empty. Returns false if it's empty.
    if trips:
        trips[node_id][''] = sum(avg_duration) // max(len(avg_duration), 1)
        trips[node_id][''] = sum(avg_length) // max(len(avg_length), 1)
        trips[node_id][''] = trip_id + 1
        trips[node_id][''] = heatmap


    return [trips, weekdays_node, hours_node]


def separateTrips(nodes: list) -> list:
    """Calls tripSeparator function and and divides the task between cores by node_id. Takes a list as an argument, which consists of dataframe divided by node id. Returns a list where the first element is the separated  trip, second element is weekdays statistics and last is hours statistics."""

    # Parallel processing
    with Pool() as p:
        results = p.map(tripSeparator, nodes)
    
    return results


def tripCleaner(nodes: list) -> dict:
    """Takes the separated list from tripSeparator function. Cleans and divides the list to weekdays, hours, and trip dictionaries, and heatmap as a list. Weekdays and hours holds the average statistics from trips."""
    
    cleaned_list = {}
    heatmap = np.zeros(getGridSize())
    
    weekdays_total = {}
    for day in range(7):
        weekdays_total[day] =  {}

    hours_total = {}
    for hour in range(24):
        hours_total[hour] =  {}
    
    for index, node in enumerate(nodes):

        # Divide the data to weekdays dictionary
        for day in node[1]:
            for key in node[1][day].keys():
                weekdays_total[day][key] += node[1][day][key]
        
        # Divide the data to hours dictionary
        for hour in node[2]:
            for key in node[2][day].keys():
                hours_total[hour][key] += node[2][hour][key]
        
        # Divide the data to trip dictionary
        node_id = list(node[0].keys())[0]
        cleaned_list[node_id] = nodes[index][0][node_id]
        heatmap += node[0][node_id]['']
           
    
    weekdays_avg = {}
    for x in range(7):
        weekdays_avg[x] =  {}
        
    # Calculate the average statistics for each weekday.
    for day in weekdays_total:
        if weekdays_total[day][] != 0:
            weekdays_avg[day][] = round(weekdays_total[day][] / weekdays_total[day][], 2)
            weekdays_avg[day][] = round(weekdays_total[day][] / weekdays_total[day][], 2)
            weekdays_avg[day][] = weekdays_total[day][]
        
    hours_avg = {}
    for x in range(24):
        hours_avg[x] =  {}
        
    # Calculate the average statistics for each hour.
    for hour in hours_total:
        if hours_total[hour][''] != 0:
            hours_avg[hour][''] = round(hours_total[hour][''] / hours_total[hour][''], 2)
            hours_avg[hour][''] = round(hours_total[hour][''] / hours_total[hour][''], 2)
            hours_avg[hour][''] = hours_total[hour]['']
        # Only include statistics from opening hours.
        if hours_avg[hour][''] == 0:
            del hours_avg[hour]
    
    return weekdays_avg, hours_avg, cleaned_list, heatmap