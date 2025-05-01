# app.py
import streamlit as st
import pandas as pd
import pgeocode
import psycopg2
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
import pydeck as pdk
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import random
import plotly.express as px
import requests
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage

# Connect to Supabase
def connect_db():
    db_url = f"postgresql+psycopg2://{os.getenv('SUPABASE_USER')}:{os.getenv('SUPABASE_PASSWORD')}@{os.getenv('SUPABASE_HOST')}:{os.getenv('SUPABASE_PORT')}/{os.getenv('SUPABASE_DB')}"
    engine = create_engine(db_url)
    return engine

# Fetch data
def get_data(query, engine):
    # engine = connect_db()
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)
    return df

# upload data to Supabase
def insert_data_to_supabase(df, table_name, engine):
    # engine = connect_db()
    try:
        with engine.connect() as conn:
            df.to_sql(table_name, conn, if_exists='append', index=False)
        st.success(f"Data uploaded to {table_name} table in Supabase.")
    except SQLAlchemyError as e:
        st.error(f"Error uploading data: {e}")


# Convert ZIP codes to latitude and longitude
def add_coordinates(df):
    nomi = pgeocode.Nominatim('us')  # US ZIP codes
    df['latitude'] = df['zipcode'].apply(lambda x: nomi.query_postal_code(str(x)).latitude)
    df['longitude'] = df['zipcode'].apply(lambda x: nomi.query_postal_code(str(x)).longitude)
    return df


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]

def df_to_json_nest(df_parent, df_child, join_key, child_key="children"):
    """
    Simplified nesting of a parent DataFrame with its child records.

    Parameters:
        df_parent (pd.DataFrame): Parent records (e.g., providers, entrepreneurs).
        df_child (pd.DataFrame): Child records (e.g., programs, needs).
        join_key (str): Column name used to join parent and child.
        child_key (str): Key under which children will be nested (default 'children').

    Returns:
        list of dicts: Nested JSON-like structure.
    """
    child_cols = [col for col in df_child.columns if col != join_key]

    parent_data = df_parent.set_index(join_key).to_dict(orient="index")

    # Fixed: exclude grouping column explicitly
    child_groups = (
        df_child.groupby(join_key)[child_cols]
        .apply(lambda x: x.to_dict(orient="records"))
        .to_dict()
    )

    nested = []
    for key, pdata in parent_data.items():
        item = {join_key: key, **pdata}
        item[child_key] = child_groups.get(key, [])
        nested.append(item)

    return nested


def summarize_user_identity_and_needs(entrepreneur_info, model):
    """
    Summarizes user identity and needs based on form data.
    
    Args:
    
    Returns:
        str: Summary paragraph describing user identity and needs.
    """
    
    # Prepare input for OpenAI summarization
    prompt = f"""
    Summarize the following user identity and needs into a concise paragraph:
    
    Entrepreneur overview: {entrepreneur_info}
    
    Use natural language to describe the user basis reported input. Stick to user provided input, don't assume or use any other information source./
    The description should reflect user's reported information accurately. Also, specify the "category" in the summary explicitly. For example, 
    say in terms of "Identity" and profile the user is XXX. In  terms of "Growth Stage" the user is XXX. In terms of "Vertical", the user is in XYZ.
    In terms of Services, the user needs XYZ.
    """
    
    # Generate summary using OpenAI model
    response = model.invoke([HumanMessage(content=prompt)])
    
    return response.content

def summarize_recommendations(entrepreneur_info, recommendations_response, model):
        # Prepare input for OpenAI summarization
    prompt = f"""
    Summarize the following recommendations into a concise paragraph. Focus on the programs and services that are most relevant to the user based on their final_socre:

    Entrepreneur overview: {entrepreneur_info}
    Recommendations: {recommendations_response}

    Use natural language to describe the recommendations. Stick to user provided input, don't assume or use any other information source.
    
    """
    
    # Generate summary using OpenAI model
    response = model.invoke([HumanMessage(content=prompt)])
    
    return response.content

def extract_unique_items(df, column_name):
    """
    Extracts unique verticals from a specified column in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The column name containing verticals (default is 'verticals').

    Returns:
        list: A sorted list of unique verticals.
    """
    # Drop NaN values and split the verticals by comma
    lst_split = [x.split(",") for x in df[column_name].dropna().unique().tolist()]
    # Flatten the list
    lst_flat = [item for sublist in lst_split for item in sublist]
    # Remove duplicates, strip whitespace, and sort the list
    return sorted(list(set([x.strip() for x in lst_flat])))