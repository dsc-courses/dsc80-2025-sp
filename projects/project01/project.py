# project.py


import pandas as pd
import numpy as np
from pathlib import Path

###
from collections import deque
from shapely.geometry import Point
###

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
pd.options.plotting.backend = 'plotly'

import geopandas as gpd

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def create_detailed_schedule(schedule, stops, trips, bus_lines):
    unique_trips = trips[trips['route_id'].isin(bus_lines)]
    unique_trips['route_id'] = pd.Categorical(unique_trips['route_id'], categories = bus_lines)

    grouped_schedule = schedule.set_index(['trip_id','stop_sequence']).reset_index()
    merged_trips_schedule = unique_trips.merge(grouped_schedule,on='trip_id')

    fully_merged = merged_trips_schedule.merge(stops,on='stop_id')

    sorted_NumStops = fully_merged.groupby('trip_id')['stop_sequence'].sum().sort_values(ascending=True)
    sorted_NumStops = sorted_NumStops.index.to_list()

    fully_merged['trip_id'] = pd.Categorical(fully_merged['trip_id'], categories = sorted_NumStops)
    
    fully_merged = fully_merged.sort_values(by=['route_id','trip_id','stop_sequence'])
    fully_merged = fully_merged.set_index('trip_id')

    return fully_merged

def visualize_bus_network(bus_df):
    # Load the shapefile for San Diego city boundary
    san_diego_boundary_path = 'data/data_city/data_city.shp'
    san_diego_city_bounds = gpd.read_file(san_diego_boundary_path)
    
    # Ensure the coordinate reference system is correct
    san_diego_city_bounds = san_diego_city_bounds.to_crs("EPSG:4326")
    
    san_diego_city_bounds['lon'] = san_diego_city_bounds.geometry.apply(lambda x: x.centroid.x)
    san_diego_city_bounds['lat'] = san_diego_city_bounds.geometry.apply(lambda x: x.centroid.y)
    
    fig = go.Figure()
    
    # Add city boundary
    fig.add_trace(go.Choroplethmapbox(
        geojson=san_diego_city_bounds.__geo_interface__,
        locations=san_diego_city_bounds.index,
        z=[1] * len(san_diego_city_bounds),
        colorscale="Greys",
        showscale=False,
        marker_opacity=0.5,
        marker_line_width=1,
    ))

    # Update layout
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center={"lat": san_diego_city_bounds['lat'].mean(), "lon": san_diego_city_bounds['lon'].mean()},
            zoom=10,
        ),
        margin={"r":0,"t":0,"l":0,"b":0}
    )

    # list of hex color strings ['#...', '#...', etc]
    # this method only gives us 10 unique colors 
    # color_palette = px.colors.qualitative.Plotly --> only has 10 unique colors
    color_palette = px.colors.qualitative.Dark24 
    color_palette = px.colors.qualitative.Plotly

    # Get all unique route_ids
    unique_routes = bus_df['route_id'].unique()

    # Build a dictionary 
    # for i, route in enumerate(unique_routes) --> i=index, route=unique_routes --> [(0,105), (1,44), etc]
    route_colors = {route: color_palette[i % len(color_palette)] for i, route in enumerate(unique_routes)}

    for route_id in bus_df['route_id'].unique():

        # create new df that is filtered by specific route_id 
        route_df = bus_df[bus_df['route_id'] == route_id]

        # get information about the specific route_id 
        fig.add_trace(go.Scattermapbox(
            lat = route_df['stop_lat'],
            lon = route_df['stop_lon'],
            mode = 'markers+lines',

            # route_colors[route_id] --> look up the specific route_id color in the dictionary 
            marker = dict(size=6, color=route_colors[route_id]),
            line = dict(color=route_colors[route_id]),

            name = f'Bus Line {route_id}',
            text = route_df['stop_name'],
            hoverinfo = 'text',
        ))
    
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center={"lat": san_diego_city_bounds['lat'].mean(), "lon": san_diego_city_bounds['lon'].mean()},
            zoom=10,
        ),
        margin={"r":0,"t":0,"l":0,"b":0}
    )

    return fig


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def find_neighbors(station_name, detailed_schedule):
    station = [station_name]

    if detailed_schedule['stop_name'].isin(station).sum() == 0: 
        return []
    
    detailed_schedule = detailed_schedule.reset_index()

    # find all unique trip_id that stop at station_name
    all_trip_id_arr = detailed_schedule[detailed_schedule['stop_name']==station_name]['trip_id'].unique()

    # filter original df by unique all_trip_id_arr array and sort each unique trip_id by ascending stop sequence 
    tripID_stopSeq_df = (detailed_schedule[detailed_schedule['trip_id'].isin(all_trip_id_arr)]
                         .sort_values(by=['trip_id','stop_sequence']))[['stop_id','stop_name','trip_id','stop_sequence']]

    # filter original df by given stop_name and select few columns desired 
    specfic_stop_df = (detailed_schedule[detailed_schedule['stop_name']==station_name]
                       [['stop_id','stop_name','trip_id','stop_sequence']])

    # adjust specific_stop_df by 'stop_sequence' column to display the sequence 
    following_stop_df = specfic_stop_df
    # increment stop_sequence num by 1 to get the "next stop"
    following_stop_df['stop_sequence'] = following_stop_df['stop_sequence'] + 1

    # merge following_stop_df and tripID_stopSeq_df to find the stop_name of the next stop given the specfied trip_id and new stop_sequence
    next_stop_name_df = following_stop_df.merge(tripID_stopSeq_df, on=['trip_id','stop_sequence'],suffixes=('_given_stop','_next_stop'))

    # goal is to just find the stop names and we may have the same stop appearing because of different route_id 
    found_next_stop = next_stop_name_df.drop_duplicates(subset=['stop_name_next_stop'])['stop_name_next_stop'].tolist()

    return found_next_stop


def bfs(start_station, end_station, detailed_schedule):
    detailed_schedule = detailed_schedule.reset_index()

    if detailed_schedule['stop_name'].isin([start_station]).sum() == 0: 
        return "Start station {start_station} not found."
    if detailed_schedule['stop_name'].isin([end_station]).sum() == 0:
        return "End station {end_station} not found."

    # A deque is a double-ended queue implementation; generalizes a stack and a queue by allowing append and pop operations from both ends of the sequence. 
    # Unlike regular lists which have O(n) time complexity for inserting or removing elements at the beginning
    # deques provide O(1) time complexity for these operations on both ends
    queue = deque([[start_station]])      # storing a list as our path 

    already_visited = set()

    while queue: # while queue not empty 
        path = queue.popleft()                                                      # path == ['stop_name'] --> notice path is a list 
        current_stop = path[-1]                                                     # looking at last element in path == 'stop_nam' --> notice path is a str

        if current_stop == end_station:                                             # basically run while loop until we get to the end_station 
            # create df --> Goal is to list the stop_name of the path in order along with lat and lon --> nothing else

            # filter df by stop_name's that are only in path
            path_df = detailed_schedule[detailed_schedule['stop_name'].isin(path)].drop_duplicates(subset='stop_name')

            # sort our df according to the order of path
            path_df['stop_name'] = pd.Categorical(path_df['stop_name'], categories = path)
            path_df = path_df.sort_values('stop_name')[['stop_name','stop_lat','stop_lon']].reset_index(drop=True)

            # add new col labeling the order number 
            path_df['stop_num'] = np.arange(1, len(path)+1)

            return path_df
    
        if current_stop not in already_visited: 
            already_visited.add(current_stop)                                       # adding str to already visited 
            next_neighbors = find_neighbors(current_stop,detailed_schedule)         # next_neighbors == [str1,str2,...]
            for neighbor in next_neighbors:                                         # neighbor = str1, neighbor = str2, ... 
                if neighbor not in already_visited:                                 # str vs str comparison
                    queue.append(path + [neighbor])                                 # path = ['stop_name'] + neighbor = ['new_stop_name'] --> ['stop_name','new_stop_name']
    
    return 


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def simulate_bus_arrivals(tau, seed=12):
    
    np.random.seed(seed) # Random seed -- do not change
    
    # The Poisson distribution describes the probability of a given number of events occurring in a fixed time period --> arrival time
    # while the exponential distribution models the time between those events --> interval(lambda = 1/tau)  

    # 6:00 AM (360 minutes) and 12:00 AM (1440 minutes) --> 1080/tau
    pt_a = 360
    pt_b = 1440
    num_buses = int((pt_b - pt_a)/tau)      # tau is a float but num_buses needs to be an int 

    sample_arrival_times = np.random.uniform(low=pt_a, high=pt_b, size=num_buses)
    sample_arrival_times = np.sort(sample_arrival_times)

    # Find arrival_time --> convert times to 24hr clock format 
    # allow 24:00 = 0 minutes and count from there 
    hour = (sample_arrival_times/60) // 1               # int portion of arrival times 
    decimal_hour = (sample_arrival_times/60) % 1        # decimal portion of arrival times 

    minute = (decimal_hour*60) // 1            
    second = (decimal_hour*60) % 1
    whole_second = np.floor(second*60)

    time_str = [f"{int(hr):02d}:{int(min):02d}:{int(sec):02d}" for hr, min, sec in zip(hour, minute, whole_second)]

    # Find Interval 
    # Key point: our array when calculating the differences will be n-1 (because pair of elements)
    time_diff = np.array([])
    time_diff = np.append(time_diff, sample_arrival_times[0] - 360)
    for i in range(1, len(sample_arrival_times)):
        time_diff = np.append(time_diff, sample_arrival_times[i] - sample_arrival_times[i - 1])

    time_diff = np.round(time_diff,decimals=2)

    final_df = pd.DataFrame()
    final_df['Arrival Time'] = time_str
    final_df['Interval'] = time_diff
    
    return final_df


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------

def convert_to_time_str(times):
    hour = (times/60) // 1               # int portion of arrival times 
    decimal_hour = (times/60) % 1        # decimal portion of arrival times 

    minute = (decimal_hour*60) // 1            
    second = (decimal_hour*60) % 1
    whole_second = np.floor(second*60)

    time_str = [f"{int(hr):02d}:{int(min):02d}:{int(sec):02d}" for hr, min, sec in zip(hour, minute, whole_second)]

    return time_str

def simulate_wait_times(arrival_times_df, n_passengers):

    start = 360         # 6AM 

    # Interval column is the difference from the prior time to the arrival time of next bus 
    # we want the accumulative running time to calculate the clock time of the each bus from 6AM
    end = max(arrival_times_df['Interval'].cumsum() + 360)       # (+ 360) to get running time from 6AM     

    # -------------------------- Passenger Arrival Time ----------------------------------------

    passenger_wait_time = np.random.rand(n_passengers) * (end - start) + start
    passenger_wait_time = np.sort(passenger_wait_time)

    str_passenger_time = convert_to_time_str(passenger_wait_time)

    # ---------------------------- Find Bus Time passenger will actually catch --------------------
    # len(str_passenger_time) > len(bus_time)

    # bus arrival times 
    bus_time = arrival_times_df['Interval'].cumsum().values + 360

    bus_index = np.array([])
    bus_time_match = np.array([])
    # check that both should be same length as passenger_wait_time by the end of loops

    for i in range(len(passenger_wait_time)):
        # handle cases where a passenger misses the very last bus
        found_match = False

        for j in range(len(bus_time)):
            if bus_time[j] >= passenger_wait_time[i]:
                bus_time_match = np.append(bus_time_match, bus_time[j])
                bus_index = np.append(bus_index, j)
                found_match = True
                break           # want to only add the first instance where if statement is true dont add every bus time after 

        if found_match == False:
            # add null values to end 
            bus_time_match = np.append(bus_time_match,np.nan)
            bus_index = np.append(bus_index, -1)

    str_bus_time_match = convert_to_time_str(bus_time_match)

    # ------------------------------------- Calculate wait time of passenger ------------------------
    # difference between the time of the bus that specifc passenger catches and passenger arrival time 
    wait_time_per = bus_time_match - passenger_wait_time
    wait_time_per = np.round(wait_time_per, decimals=2)

    simu_df = pd.DataFrame(
        {'Passenger Arrival Time': str_passenger_time,
        'Bus Arrival Time': str_bus_time_match,
        'Bus Index': bus_index,
        'Wait Time': wait_time_per,}
    )

    return simu_df

def visualize_wait_times(wait_times_df, timestamp):
    # pd.Timestamp('13:00:00') means we want to get the waiting times between 13:00:00 to 14:00:00 (exclusive)
    # means we need access to Passenger and Bus Arrival Time for x axis 

    # ----------------------------------------- Convert Col Str to Time Objects --------------------------------------

    # col: ['Passenger Arrival Time','Bus Arrival Time'] are str and not time objects 
    time_obj_df = wait_times_df.copy()
    time_obj_df['Passenger Arrival Time'] = pd.to_datetime(wait_times_df['Passenger Arrival Time'])
    time_obj_df['Bus Arrival Time'] = pd.to_datetime(wait_times_df['Bus Arrival Time'])

    # find interval 
    start = timestamp
    end = timestamp + pd.Timedelta(hours=1)

    # ----------------------------------------- Find Data that is in 1hr Interval --------------------------------------

    # take from our time_obj_df the arrival times that fall between start <= arrival time <= end 

    # series of T/F where our Passenger Arrival Time falls between interval 
    specific_block_series = (time_obj_df['Passenger Arrival Time'] >= timestamp) & (time_obj_df['Passenger Arrival Time'] < end)
    data_in_interval_df = time_obj_df[specific_block_series].copy()

    # --------------------------------------- Get Time Object in Minutes (x-axis) --------------------------------------

    # time from start to actual Passenger Arrival time 
    # example: 13:02:00 - 13:00:00 = 2 minutes
    # possible bc column and timestap are both datetime objects from pd.Timestamp
    data_in_interval_df['x_passenger'] = (data_in_interval_df['Passenger Arrival Time'] - timestamp) 
    data_in_interval_df['x_passenger'] = (data_in_interval_df['x_passenger']).dt.total_seconds() / 60      # allows us to get time obj in seconds but / 60 bc we want minutes

    data_in_interval_df['x_bus'] = (data_in_interval_df['Bus Arrival Time'] - timestamp).dt.total_seconds() / 60

    # -------------------------------------------- Visualize -----------------------------------------------
    # red vertical lines from passenger arrival time to x axis 
    vert_wait_lines = go.Scatter(     # requires access from stimulate_wait_times['wait_time']
        x=sum([[x, x, None] for x in data_in_interval_df['x_passenger']], []),
        y=sum([[0, y, None] for y in data_in_interval_df['Wait Time']], []),
        mode='lines',
        line=dict(color='red', dash='dot'),
        showlegend=False
    )

    # (passenger arrival time, passenger wait time) : red dot 
    passenger_pts = go.Scatter(
        x=data_in_interval_df['x_passenger'],
        y=data_in_interval_df['Wait Time'],
        mode='markers',
        marker=dict(color='red'),
        name='Passengers'
    )

# (buss arrival time, 0) : blue dot 
    bus_pts = go.Scatter(
        x=data_in_interval_df['x_bus'],
        y=[0] * len(data_in_interval_df),
        mode='markers',
        marker=dict(color='blue'),
        name='Buses'
    )

    layout = go.Layout(
        title='Passenger Wait Times in a 60-Minute Block',
        xaxis=dict(title='Time (minutes) within the block'),
        yaxis=dict(title='Wait Time (minutes)')
    )
    fig = go.Figure(data=[vert_wait_lines, passenger_pts, bus_pts], layout=layout)
    return fig
