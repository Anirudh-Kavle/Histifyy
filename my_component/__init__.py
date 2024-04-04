import streamlit as st
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import streamlit.components.v1 as components

# Declare the component
my_component = components.declare_component(
    "my_component",
    url="http://localhost:3001"  # Make sure the URL is correct
)

# Render the component
my_component()


