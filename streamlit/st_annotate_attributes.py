import glob
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd

import streamlit as st
from utils import (ConsistencyChecker, get_default, inverse_parse_format_dtype,
                   parse_format_dtype)

data_dir_list = os.listdir(os.environ["MA_DATA_DIR"])

data_dir_name = st.sidebar.selectbox("Dataset", data_dir_list)
data_dir = os.path.join(os.environ["MA_DATA_DIR"], data_dir_name)
lanes_attributes_file = os.path.join(data_dir, "LanesAttributes.json")
offroad_attributes_file = os.path.join(data_dir, "OffroadAttributes.json")

fill_forms = st.sidebar.checkbox("Fill fields with defaults")


# Load all road element geometries
intersections = gpd.GeoDataFrame.explode(
    gpd.read_file(os.path.join(data_dir, "Intersections.gpkg")), index_parts=False
)

offroad = gpd.GeoDataFrame.explode(
    gpd.read_file(os.path.join(data_dir, "Offroad.gpkg")), index_parts=False
)

crosswalks = gpd.GeoDataFrame.explode(
    gpd.read_file(os.path.join(data_dir, "Crosswalks.gpkg")), index_parts=False
)

lanes = gpd.GeoDataFrame.explode(
    gpd.read_file(os.path.join(data_dir, "Lanes.gpkg")), index_parts=False
)

terminal = gpd.GeoDataFrame.explode(
    gpd.read_file(os.path.join(data_dir, "Terminal.gpkg")), index_parts=False
)

geometry_dict = {
    "lanes": lanes,
    "intersections": intersections,
    "offroad": offroad,
    "crosswalks": crosswalks,
    "terminal": terminal,
}

# Load all attribute forms
with open(lanes_attributes_file, "r") as f:
    lanes_attributes = json.load(f)
with open(offroad_attributes_file, "r") as f:
    offroad_attributes = json.load(f)

attributes_fps = {"lanes": lanes_attributes_file, "offroad": offroad_attributes_file}

attributes_dict = {
    "lanes": lanes_attributes,
    "offroad": offroad_attributes,
}

# Annotation selection
annotations_names = list(geometry_dict.keys())
element_to_annotate = st.sidebar.selectbox("Annotate element", annotations_names)
element_geometry = geometry_dict[element_to_annotate]
element_attributes = attributes_dict[element_to_annotate]

feature_ids = element_geometry["element_id"].tolist()
chosen_id = st.sidebar.selectbox("Feature ID", sorted(feature_ids))
chosen_id = int(chosen_id)

chosen_geometry = element_geometry[element_geometry["element_id"] == chosen_id]

attribute_format = [
    ("lane_id", "int64"),
    ("from_object", "str"),
    ("to_object", "str"),
    ("connects_to", "list(str)"),
    ("successors", "list(str)"),
    ("predecessors", "list(str)"),
]

checker = ConsistencyChecker(attribute_format)

attributes = {}
# Feature attribute form
if str(chosen_id) in element_attributes.keys():
    # pre-fill form with entries
    attributes = lanes_attributes[str(chosen_id)]
    st.write("Found entry:")
    st.write(attributes)


for key, dtype in attribute_format:
    casting_func = parse_format_dtype(dtype)
    inv_casting_func = inverse_parse_format_dtype(dtype)
    if key == "lane_id":
        value = int(chosen_id)
    else:
        value = st.text_input(key, value=get_default(attributes, key, inv_casting_func))
        value = None if value == "" else casting_func(value)

    attributes[key] = value

st.write("Attribute dict:")
st.write(attributes)

if st.button("Save form") or st.sidebar.button("Save form", key="gsjdk"):
    # run consistency check and update attributes dict
    element_attributes = checker.update_neighbours(
        str(chosen_id), element_attributes, feature_ids
    )

    element_attributes[str(chosen_id)] = attributes
    fp = attributes_fps[element_to_annotate]
    with open(fp, "w") as f:
        json.dump(element_attributes, f, indent=2)

    st.write("Form saved")
    st.sidebar.write("Form saved")
