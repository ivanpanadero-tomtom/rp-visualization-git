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

def provider_char_d(res):
    return res['responses'][0]['lat'], res['responses'][0]['lon']

def provider_latlon_(res):
    try:
        return res['responses'][0]['lat'], res['responses'][0]['lon']
    except (KeyError, IndexError, TypeError) as e:
        # Log the error for debugging if needed
        return []  # Return an empty list on error

def max_distance(centroid, markers):
    max_dist = 0
    # Compare every pair of markers
    for i in markers:
            dist = geodesic(centroid, i).km
            if dist > max_dist:
                max_dist = dist

    return max_dist*1000  # Return the distance in meters

# Function to calculate bounding box from a point and a distance (in meters)
def calculate_bounds(lat, lon, distance_meters):
    # Approximate the distance in degrees (1 degree of latitude is about 111 km)
    lat_offset = distance_meters / 111320  # 1 degree of latitude ~ 111320 meters
    lon_offset = distance_meters / (40008000 * (1 / 360)) * (1 / (111320 * 2))  # Roughly adjust for longitude

    # Calculate the bounding box
    min_lat = lat - lat_offset
    max_lat = lat + lat_offset
    min_lon = lon - lon_offset
    max_lon = lon + lon_offset

    return [[min_lat, min_lon], [max_lat, max_lon]]

# Select a row to visualize (Change index as needed)

# Example: Load your DataFrame (adjust path as needed)
df_pandas = pd.read_parquet('data_esp')

# Extract relevant fields
df_pandas['num_reference_routing_points'] = df_pandas["reference_routing_points"].apply(len)
df_pandas['num_provider_routing_points'] = df_pandas["provider_routing_points"].apply(len)

# Create a list of names with their corresponding details
names_with_info = [
    f"{name} - {category} - ({num_ref} - {num_provider}) - RPPA = {rppa}"
    for name, category, num_ref, num_provider, rppa in zip(
        df_pandas["name"], 
        df_pandas["category_name"], 
        df_pandas["num_reference_routing_points"], 
        df_pandas["num_provider_routing_points"], 
        df_pandas["rpav_matching"].apply(lambda x: x['fields']['rppa'])
    )
]

# Create a dictionary mapping the formatted names to their row indices
name_to_index = {info: idx for idx, info in enumerate(names_with_info)}

# Create a dropdown (selectbox) with the formatted names and categories
st.write("Select a POI to visualize:")
seleccion = st.selectbox("POI name - POI Category - (#RPs Reference - #RPs Provider) - RPPA = X:", names_with_info)

# Get the row index associated with the selected formatted name
selected_row_index = name_to_index[seleccion]
row = df_pandas.iloc[selected_row_index]
# Get the row index associated with the selected formatted name


rppa = row['rpav_matching']['fields']['rppa']
reference_routing_points =  row["reference_routing_points"]
provider_routing_points = row["provider_routing_points"]
poi_characteristic_distance = row['rpav_matching']['fields']['poi_characteristic_distance']
assignation = row['rpav_matching']['fields']['assignation']
reference_latlon = (float(row['ref_lat']), float(row['ref_lon']))
provider_latlon = provider_latlon_(json.loads(row['provider_response']))  

st.text(f"RPPA: {rppa}")

map_centroid = reference_latlon

width, height = 800, 600 
m = folium.Map(location=map_centroid, zoom_start=17, tiles='openstreetmap', width=width, height=height)

folium.Marker(location= reference_latlon, icon=folium.Icon(color='black', icon = "")).add_to(m)

if provider_latlon and isinstance(provider_latlon, tuple) and len(provider_latlon) == 2:
    folium.Marker(location=provider_latlon, icon=folium.Icon(color='red', icon="")).add_to(m)

markers = [reference_latlon, provider_latlon]

for rp in provider_routing_points:
    folium.Circle(location=rp, radius=0.7*poi_characteristic_distance, color="red").add_to(m)
    folium.CircleMarker(location=rp, radius=4, color="red", fill=False, fill_color="red", fill_opacity=1).add_to(m)

    markers.append(rp)
    folium.PolyLine(locations=[rp, provider_latlon], color="red", weight=2, opacity=1, dashArray="5, 5").add_to(m)

if rppa > 0:
    for asign in assignation:
        if geodesic(reference_routing_points[asign[0]], provider_routing_points[asign[1]]).m < 0.7*poi_characteristic_distance:
            folium.PolyLine(locations=[reference_routing_points[asign[0]], provider_routing_points[asign[1]]], color="green", weight=4, opacity=1).add_to(m)

for rp in reference_routing_points:
    folium.CircleMarker(location=rp, radius=4, color="black", fill=True, fill_color="black", fill_opacity=1).add_to(m)
    markers.append(rp)
    folium.PolyLine(locations=[rp, reference_latlon], color="black", weight=2, opacity=1, dashArray="5, 5").add_to(m)


    
bounds = calculate_bounds(map_centroid[0], map_centroid[1], 1.5*max_distance(map_centroid, markers))
m.fit_bounds(bounds)

# Render Map as Image
img_data = m._to_png(0.5)
img = Image.open(io.BytesIO(img_data))
rppa = row['rpav_matching']['fields']['rppa']
def add_frame(img, rppa, frame_width=10):
    """
    Adds a frame to the given image.
    
    :param img: PIL.Image object
    :param frame_width: Width of the frame in pixels
    :return: New PIL.Image object with the frame
    """
    # Define the color of the frame (RGB green)
    frame_color = (int(255*(1-rppa)), int(rppa*255), 0)
    
    # Add the frame using ImageOps.expand
    framed_img = ImageOps.expand(img, border=frame_width, fill=frame_color)
    
    return framed_img

# Display in Streamlit

img = add_frame(img, rppa)
st.image(img, caption=seleccion, use_container_width=True)
########

def create_legend():
    legend_html = """
    <div style="position: fixed; 
                bottom: 10px; left: 10px; width: 200px; height: 160px; 
                background-color: white; border: 2px solid black; z-index: 9999; 
                padding: 10px; font-size: 14px;">
        <strong>Legend</strong><br>
        <i style="background-color:black; width: 20px; height: 20px; display: inline-block; margin-right: 5px; border-radius: 90%;"></i> Reference RPs<br>
        <i style="background-color:red; width: 20px; height: 20px; display: inline-block; margin-right: 5px; border-radius: 90%;"></i> Provider RPs<br>
        <i style="background-color:green; width: 20px; height: 5px; display: inline-block; margin-right: 5px;"></i> Assignation<br>
    </div>
    </div>
    """
    return legend_html

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

# Add custom CSS to adjust the position of the image in the sidebar
st.markdown("""
    <style>
        .css-1d391kg {
            position: absolute;
            bottom: 20px;  /* Adjust to move the image to the bottom */
            width: 200px;  /* Adjust width as needed */
            left: 50%;  /* Center the image horizontally */
            transform: translateX(-50%);  /* Perfect centering */
        }
    </style>
""", unsafe_allow_html=True)