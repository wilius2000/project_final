'''
Name:       Vilius Vysniauskas
CS230:      Section 1
Dataset:    Shipwrecks
URL:        (Yet to be posted)

Description:
            this code analyzes the ShipwreckDatabase dataset. It explores:
            the relationship between the material ship was built from and cause of sinkage;
            highest amount of casualties, most common vessel types, ship value and cargo value of ships;
            provides a map with all of the shipwrecks throughout the years, which allows the user to see how shipwrecks
            decreased towards the end of 20th century;
            provides a stacked chart of different types of materials used by the main manufacturing locations;
            provides pie charts used to analyze different types of cargos used by 5 main vessel types.
'''
import pandas as pd
import folium
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import plotly.express as px

data = pd.read_csv('ShipwreckDatabase.csv')
# tabs represent different tabs on Streamlit
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Most frequent cause of shipwreck per material of ship", "Lives lost, "
            "ship value, cargo value per vessel type", "Map of shipwrecks across the years", "Top 10 most common "
            "ship producers and materials of ships they made", "Cargos carried per 5 most popular vessel types"])
def get_cause_of_loss_by_construction(construction):
    df = data[data['CONSTRUCTION'] == construction]
    cause_counts = df['CAUSE OF LOSS'].value_counts()
    most_likely_cause = cause_counts.index[0]
    return most_likely_cause


def vessels_top(df):
    # Convert columns to numeric data types, ignoring non-numeric values
    df['LIVES LOST'] = pd.to_numeric(df['LIVES LOST'], errors='coerce').round(0)
    df['SHIP VALUE'] = pd.to_numeric(df['SHIP VALUE'], errors='coerce').round(2)
    df['CARGO VALUE'] = pd.to_numeric(df['CARGO VALUE'], errors='coerce').round(2)

    grouped = df.groupby('VESSEL TYPE').agg({'LIVES LOST': lambda x: x.sum(numeric_only=True),
                                            'SHIP VALUE': lambda x: x.mean(numeric_only=True),
                                             'CARGO VALUE': lambda x: x.mean(numeric_only=True),
                                             'VESSEL TYPE': lambda x: x.count()})
# top 5 most common vessel types
    top_5_vessels = df['VESSEL TYPE'].value_counts().head(5).index.tolist()
    filtered_data = grouped.loc[top_5_vessels]
    return filtered_data
def generate_pie_chart(vessel_type):
    selected_data = data[data['VESSEL TYPE'] == vessel_type]
# how much of what cargo each vessel type carries
    cargo_counts = selected_data['NATURE OF CARGO'].value_counts()
    fig = px.pie(cargo_counts, values=cargo_counts.values, names=cargo_counts.index)
    st.plotly_chart(fig)


def display_ships_by_location_and_material(data):
    location_counts = data['WHERE BUILT'].value_counts()
    construction_counts = data['CONSTRUCTION'].value_counts()

    main_locations = data.groupby('CONSTRUCTION')['WHERE BUILT'].apply(
        lambda x: x.value_counts().idxmax() if not pd.isnull(x).all() else 'unknown')

    main_location_counts = data['WHERE BUILT'].value_counts()
    main_location_counts = main_location_counts.sort_values(ascending=False)
    top_10_locations = main_location_counts.head(10).index
    top_10_data = data[data['WHERE BUILT'].isin(top_10_locations)]

    grouped_data = top_10_data.groupby(['WHERE BUILT', 'CONSTRUCTION']).size().unstack()

    colors = ['sandybrown', 'yellow', 'darkred', 'red', 'brown']
    fig, ax = plt.subplots(figsize=(10, 6))
    grouped_data.plot(kind='bar', stacked=True, color=colors, ax=ax)

    plt.xlabel('Location')
    plt.ylabel('Number of Ships')
    plt.title('Number of Ships Produced by Location and Material')
    plt.gca().set_facecolor('black')
    plt.xticks(color='black')
    plt.yticks(color='black')
    plt.title('Number of Ships Produced by Location and Material', color='black')
    plt.xlabel('Location', color='black')
    plt.ylabel('Number of Ships', color='black')
    legend = plt.legend(title='Construction Material', facecolor='black', edgecolor='white')
    plt.setp(legend.get_title(), color='white')
    for text in legend.get_texts():
        plt.setp(text, color='white')

    st.pyplot(fig)


# 1
with tab1:
    st.header('Relationship between Cause of Loss and Construction')
    construction = st.selectbox('Select a construction material', options=data['CONSTRUCTION'].unique())
    if st.button('Get most likely cause of loss'):
        result = get_cause_of_loss_by_construction(construction)
        st.write(f'The most likely cause of loss for ships made of {construction} is {result}.')
# 2
with tab2:
    vessel_data = vessels_top(data)
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams.update({'font.size': 14})

    fig, ax1 = plt.subplots(figsize=(12, 8), facecolor='black')
    num_extra_bars = 2
    bar_width = 0.8 / (num_extra_bars + 2)
# set axis length
    x_pos = np.arange(len(vessel_data))

# color codes retrieved from chatGPT
    ax1.set_title('Shipwreck Data by Vessel Type', color='white')
    ax1.bar(x_pos - num_extra_bars * bar_width, vessel_data['LIVES LOST'], width=bar_width, align='edge', label='LIVES LOST', color='#800000')
    ax1.bar(x_pos - (num_extra_bars - 1) * bar_width, vessel_data['VESSEL TYPE'], width=bar_width, align='edge', label='VESSEL TYPE COUNT', color='#FF0000')
    ax1.set_ylabel('Count / Lives Lost', color='white')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(vessel_data.index)

# learned how to install a second axis through chatGPT
# 2nd y axis
    ax2 = ax1.twinx()
    ax2.bar(x_pos, vessel_data['SHIP VALUE'], width=bar_width, align='edge', label='SHIP VALUE', color='#FEC547')
    ax2.bar(x_pos + bar_width, vessel_data['CARGO VALUE'], width=bar_width, align='edge', label='CARGO VALUE', color='#FFFF00')
    ax2.set_ylabel('Average Value ($)', color='white')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2)

    st.pyplot(fig)
# 3
with tab3:
    data['LATITUDE_BACKUP'] = pd.to_numeric(data['LATITUDE_BACKUP'], errors='coerce')
    data['LONGITUDE_BACKUP'] = pd.to_numeric(data['LONGITUDE_BACKUP'], errors='coerce')

    data = data.dropna(subset=['LATITUDE_BACKUP', 'LONGITUDE_BACKUP'])

    st.title("Map of Shipwrecks")
    st.write("This map will show shipwrecks and provide information on hover.")

    selected_year = st.slider("Select a Year: ", min_value=int(data['YEAR'].min()), max_value=int(data['YEAR'].max()),
                              step=1)

    filtered_data = data[data['YEAR'] == selected_year]

    shipwreck_count = filtered_data.shape[0]

    st.write(f"Shipwrecks in {selected_year}: {shipwreck_count}")

    m = folium.Map(location=[data['LATITUDE_BACKUP'].mean(), data['LONGITUDE_BACKUP'].mean()])
    marker_cluster = MarkerCluster().add_to(m)

    for index, row in filtered_data.iterrows():
        lat = row['LATITUDE_BACKUP']
        lon = row['LONGITUDE_BACKUP']
        name = row["SHIP'S NAME"]

        marker = folium.Marker(location=[lat, lon], popup=name)
        marker.add_to(marker_cluster)

    max_lat = filtered_data['LATITUDE_BACKUP'].max()
    min_lat = filtered_data['LATITUDE_BACKUP'].min()
    max_lon = filtered_data['LONGITUDE_BACKUP'].max()
    min_lon = filtered_data['LONGITUDE_BACKUP'].min()

    if not filtered_data.empty:
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    if shipwreck_count > 0:
        folium_static(m)
    else:
        st.warning("No shipwrecks found for the selected year.")

# 4
with tab4:



    display_ships_by_location_and_material(data)
# 5
with tab5:
    filtered_data_5 = data.dropna(subset=['NATURE OF CARGO'])
    unique_cargos = filtered_data_5['NATURE OF CARGO'].unique()
# dictionary with vessel types and cargos they carried
    vessel_cargo_dict = {}
    for cargo in unique_cargos:
        vessel_types = filtered_data_5[filtered_data_5['NATURE OF CARGO'] == cargo]['VESSEL TYPE'].unique()
        vessel_cargo_dict[cargo] = vessel_types

    top_5_vessel_types = data['VESSEL TYPE'].value_counts().nlargest(5).index

    st.title("Cargo carried per vessel types.")
    vessel_select = st.selectbox("Select Vessel Type:", options=top_5_vessel_types)

# react to the selection of user
    if vessel_select:
        generate_pie_chart(vessel_select)
