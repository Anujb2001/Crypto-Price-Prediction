# import the liberary
import streamlit as st
import task7 as gui
import task1 as downloader
# set the page layout and title
st.set_page_config(layout="wide", page_title="Intelligent Coin Trading (ICT) Platform")
# create the coins and model folder if not present
downloader.create_coins_and_model_dir()
# load the side menu bar
selected_option = gui.load_side_menu()
# load the pages based on selected option
gui.display_download_page(selected_option)
gui.display_grouping_page(selected_option)
gui.display_correlation_page(selected_option)
gui.display_EDA_page(selected_option)
gui.display_prediction_page(selected_option)
gui.display_forecasting_page(selected_option)



 