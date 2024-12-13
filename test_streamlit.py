import streamlit as st
import folium
from streamlit_folium import st_folium
import json
from geopy.distance import geodesic
import io
from PIL import Image, ImageOps
import pandas as pd

# Streamlit App Title
st.title("Visualize Geospatial Data with Streamlit and Folium")

df_pandas = pd.read_csv('esp_rppa.csv')

# Helper Functions

def provider_latlon_(row):
    res = json.loads(row['provider_response'])
    return res['responses'][0]['lat'], res['responses'][0]['lon']

def max_distance(centroid, markers):
    max_dist = 0
    # Compare every pair of markers
    for i in markers:
            dist = geodesic(centroid, i).km
            if dist > max_dist:
                max_dist = dist

    return max_dist*1000  # Return the distance in meters

def calculate_bounds(lat, lon, distance_meters):
    """Calculate a bounding box around a point."""
    lat_offset = distance_meters / 111320  # 1 degree latitude ~ 111.32 km
    lon_offset = distance_meters / (40008000 * (1 / 360)) * (1 / (111320 * 2))
    return [[lat - lat_offset, lon - lon_offset], [lat + lat_offset, lon + lon_offset]]

# Select a row to visualize (Change index as needed)


# Create a list of names and a corresponding dictionary mapping names to their row indices
nombres = df_pandas["name"].tolist()
name_to_index = {name: idx for idx, name in enumerate(nombres)}

# Create a dropdown (selectbox) with the names
seleccion = st.selectbox("Select POI:", nombres)

# Get the row index associated with the selected name
selected_row_index = name_to_index[seleccion]
row = df_pandas.iloc[selected_row_index]

poi_characteristic_distance = row['rpav_matching']['fields']['poi_characteristic_distance']
reference_latlon = (float(row['ref_lat']), float(row['ref_lon']))
provider_latlon = provider_latlon_(row)  

map_centroid = reference_latlon

width, height = 800, 600 
m = folium.Map(location=map_centroid, zoom_start=17, tiles='openstreetmap', width=width, height=height)

folium.Marker(location= reference_latlon, icon=folium.Icon(color='black', icon = "")).add_to(m)
folium.Marker(location= provider_latlon, icon=folium.Icon(color='red', icon = "")).add_to(m)

markers = [reference_latlon, provider_latlon_coords]

for rp in row["provider_routing_points"]:
    folium.Circle(location=rp, radius=0.7*poi_characteristic_distance, color="red").add_to(m)
    folium.CircleMarker(location=rp, radius=4, color="red", fill=False, fill_color="red", fill_opacity=1).add_to(m)

    markers.append(rp)
    folium.PolyLine(locations=[rp, provider_latlon], color="red", weight=2, opacity=1, dashArray="5, 5").add_to(m)

for asign in row['rpav_matching']['fields']['assignation']:
    folium.PolyLine(locations=[row["provider_routing_points"][asign[1]], row["reference_routing_points"][asign[0]]], color="green", weight=4, opacity=1).add_to(m)


for rp in row["reference_routing_points"]:
    folium.CircleMarker(location=rp, radius=4, color="black", fill=True, fill_color="black", fill_opacity=1).add_to(m)
    markers.append(rp)
    folium.PolyLine(locations=[rp, reference_latlon], color="black", weight=2, opacity=1, dashArray="5, 5").add_to(m)


    
bounds = calculate_bounds(map_centroid[0], map_centroid[1], 1.5*max_distance(map_centroid, markers))
m.fit_bounds(bounds)

#st_folium(m, width=700, height=500)


# Render Map as Image
img_data = m._to_png(0.5)
img = Image.open(io.BytesIO(img_data))
rppa = int(row['rpav_matching']['fields']['rppa'])
def add_frame(img, rppa, frame_width=10):
    """
    Adds a frame to the given image.
    
    :param img: PIL.Image object
    :param frame_width: Width of the frame in pixels
    :return: New PIL.Image object with the frame
    """
    # Define the color of the frame (RGB green)
    frame_color = (255*(1-rppa), rppa*255, 0)
    
    # Add the frame using ImageOps.expand
    framed_img = ImageOps.expand(img, border=frame_width, fill=frame_color)
    
    return framed_img

# Display in Streamlit

img = add_frame(img, rppa)
st.image(img, caption="Generated Folium Map", use_container_width=True)
########