import streamlit as st
import folium
from streamlit_folium import st_folium
import json
from geopy.distance import geodesic
import io
from PIL import Image, ImageOps
import pandas as pd

# Streamlit App Title
st.title("Routing Points (RP) Visualization")

# Example list of options for the dropdown (desplegable)
countries = ['Spain', 'Netherlands', 'Great Britain']

# Create a dropdown in the sidebar to select a country
selected_option = st.sidebar.selectbox("Select a country:", countries)

# Function to load data based on the selected country
@st.cache_data
def load_data(country):
    if country == 'Spain':
        return pd.read_parquet('data_esp')
    elif country == 'Netherlands':
        return pd.read_parquet('data_nld')
    elif country == 'Great Britain':
        return pd.read_parquet('data_gbr')

# Load the appropriate dataset
df_pandas = load_data(selected_option)

# Helper Functions
def provider_latlon_(res):
    try:
        return res['responses'][0]['lat'], res['responses'][0]['lon']
    except (KeyError, IndexError, TypeError):
        return []  # Return an empty list on error

def max_distance(centroid, markers):
    return max(geodesic(centroid, marker).km for marker in markers) * 1000  # Return the distance in meters

def calculate_bounds(lat, lon, distance_meters):
    lat_offset = distance_meters / 111320
    lon_offset = distance_meters / (40008000 * (1 / 360)) * (1 / (111320 * 2))
    return [[lat - lat_offset, lon - lon_offset], [lat + lat_offset, lon + lon_offset]]

# Preprocess data to prepare options for POI selection
def prepare_poi_options(data):
    data['num_reference_routing_points'] = data["reference_routing_points"].apply(len)
    data['num_provider_routing_points'] = data["provider_routing_points"].apply(len)

    # Create a list of names with their corresponding details
    names_with_info = [
        f"{name} - {category} - [{num_ref}, {num_provider}] - RPPA = {rppa}"
        for name, category, num_ref, num_provider, rppa in zip(
            data["name"], 
            data["category_name"], 
            data["num_reference_routing_points"], 
            data["num_provider_routing_points"], 
            data["rpav_matching"].apply(lambda x: x['fields']['rppa'])
        )
    ]

    # Create a dictionary mapping the formatted names to their row indices
    name_to_index = {info: idx for idx, info in enumerate(names_with_info)}
    return names_with_info, name_to_index

# Prepare POI options based on the selected dataset
names_with_info, name_to_index = prepare_poi_options(df_pandas)

# Create a dropdown (selectbox) with the formatted names and categories
st.write("Select a POI to visualize:")
seleccion = st.selectbox("POI name - POI Category - [#RPs Reference, #RPs Provider] - RPPA = X:", names_with_info)

# Get the row index associated with the selected formatted name
selected_row_index = name_to_index[seleccion]
row = df_pandas.iloc[selected_row_index]

# Extract relevant data from the selected row
rppa = row['rpav_matching']['fields']['rppa']
reference_routing_points = row["reference_routing_points"]
provider_routing_points = row["provider_routing_points"]
poi_characteristic_distance = row['rpav_matching']['fields']['poi_characteristic_distance']
assignation = row['rpav_matching']['fields']['assignation']
reference_latlon = (float(row['ref_lat']), float(row['ref_lon']))
provider_latlon = provider_latlon_(json.loads(row['provider_response']))

# Display RPPA value
# Define RPPA color based on rppa value
rppa_color = f"rgb({int(255 * (1 - rppa))}, {int(rppa * 255)}, 0)"

# Display centered and styled RPPA text
st.markdown(
    f"""
    <div style="
        text-align: center;
        background-color: {rppa_color};
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-size: 24px;
        font-weight: bold;">
        RPPA: {rppa}
    </div>
    """,
    unsafe_allow_html=True
)

# Create the map

map_centroid = reference_latlon
width, height = 800, 600 
m = folium.Map(location=map_centroid, zoom_start=17, tiles='openstreetmap', width=width, height=height)

folium.Marker(location=reference_latlon, icon=folium.Icon(color='black', icon="")).add_to(m)

if provider_latlon and isinstance(provider_latlon, tuple) and len(provider_latlon) == 2:
    folium.Marker(location=provider_latlon, icon=folium.Icon(color='red', icon="")).add_to(m)

markers = [reference_latlon, provider_latlon]

for rp in provider_routing_points:
    folium.Circle(location=rp, radius=0.7 * poi_characteristic_distance, color="red").add_to(m)
    folium.CircleMarker(location=rp, radius=4, color="red", fill=False, fill_color = 'red', fill_opacity = 1).add_to(m)
    folium.PolyLine(locations=[rp, provider_latlon], color="red", weight=2, dashArray="5, 5").add_to(m)
    markers.append(rp)

if rppa > 0:
    for asign in assignation:
        if geodesic(reference_routing_points[asign[0]], provider_routing_points[asign[1]]).m < 0.7 * poi_characteristic_distance:
            folium.PolyLine(locations=[reference_routing_points[asign[0]], provider_routing_points[asign[1]]], color="green", weight=4).add_to(m)

for rp in reference_routing_points:
    folium.CircleMarker(location=rp, radius=4, color="black", fill=False, fill_color = 'black', fill_opacity = 1).add_to(m)
    folium.PolyLine(locations=[rp, reference_latlon], color="black", weight=2, dashArray="5, 5").add_to(m)

bounds = calculate_bounds(map_centroid[0], map_centroid[1], 1.5 * max_distance(map_centroid, markers))
m.fit_bounds(bounds)

# Render the map directly in Streamlit
st_folium(m, width=800, height=600)
# Display the legend

for i in range(25):
    st.sidebar.markdown("""
        <style>
            .sidebar .sidebar-content {
                border-left: 20px solid #000000;  # Adjust the color and width as needed
                padding-left: 10px;  # Optional: Adds space after the line
            }
        </style>
    """, unsafe_allow_html=True)

st.sidebar.image('legend.png', caption='Legend', width=200)