# functions.py

import os
import pandas as pd
import pgeocode
import requests
import streamlit as st
import pydeck as pdk
import plotly.express as px
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from langchain_core.messages import HumanMessage
from math import radians, sin, cos, sqrt, atan2


def get_secret(key, fallback=None):
    """
    Retrieves a secret from Streamlit's secrets or environment variables.
    """
    try:
       return os.getenv(key) or st.secrets.get(key) or fallback
    except Exception:
        return os.getenv(key) or fallback

# Connect to Supabase
def connect_db():
    """
    Establishes a connection to the Supabase PostgreSQL database.
    
    Returns:
        SQLAlchemy engine object for database operations
    """
    db_url = f"postgresql+psycopg2://{get_secret('SUPABASE_USER')}:{get_secret('SUPABASE_PASSWORD')}@{get_secret('SUPABASE_HOST')}:{get_secret('SUPABASE_PORT')}/{get_secret('SUPABASE_DB')}"
    engine = create_engine(db_url)
    return engine

# Fetch data
def get_data(query, engine, params=None):
    """
    Fetches data from the database using the provided query.
    
    Args:
        query: SQL query string
        engine: SQLAlchemy engine
        params: Optional parameters for the query
        
    Returns:
        DataFrame containing query results
    """
    try:
        with engine.connect() as conn:
            if params:
                df = pd.read_sql_query(query, conn, params=params)
            else:
                df = pd.read_sql_query(query, conn)
            return df
    except SQLAlchemyError as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

# Upload data to Supabase
def insert_data_to_supabase(df, table_name, engine):
    """
    Uploads data to a Supabase table.
    
    Args:
        df: DataFrame containing data to upload
        table_name: Target table name
        engine: SQLAlchemy engine
    """
    try:
        with engine.connect() as conn:
            df.to_sql(table_name, conn, if_exists='append', index=False)
            st.success(f"Data uploaded to {table_name} table in Supabase.")
    except SQLAlchemyError as e:
        st.error(f"Error uploading data: {e}")

# Convert ZIP codes to latitude and longitude
def add_coordinates(df):
    """
    Adds latitude and longitude coordinates based on ZIP codes.
    
    Args:
        df: DataFrame containing a 'zipcode' column
        
    Returns:
        DataFrame with added 'latitude' and 'longitude' columns
    """
    nomi = pgeocode.Nominatim('us')  # US ZIP codes
    
    # Process in batches to avoid performance issues
    unique_zipcodes = df['zipcode'].unique()
    zipcode_coords = {}
    
    for zipcode in unique_zipcodes:
        try:
            result = nomi.query_postal_code(str(zipcode))
            zipcode_coords[zipcode] = (result.latitude, result.longitude)
        except Exception:
            zipcode_coords[zipcode] = (None, None)
    
    df['latitude'] = df['zipcode'].map(lambda x: zipcode_coords.get(x, (None, None))[0])
    df['longitude'] = df['zipcode'].map(lambda x: zipcode_coords.get(x, (None, None))[1])
    
    return df

def hex_to_rgb(hex_color):
    """
    Converts a hex color code to RGB values.
    
    Args:
        hex_color: Hex color code (e.g., '#FF5733')
        
    Returns:
        List of RGB values [r, g, b]
    """
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

def estimate_zipcode_distance(zip1, zip2, country="US"):
    """
    Estimate the distance in kilometers between two zip codes.
    
    Args:
        zip1 (str): First ZIP code
        zip2 (str): Second ZIP code
        country (str): Country code (default: "US")

    Returns:
        float: Distance in kilometers (or None if ZIPs invalid)
    """
    nomi = pgeocode.Nominatim(country)

    loc1 = nomi.query_postal_code(str(zip1))
    loc2 = nomi.query_postal_code(str(zip2))

    # Check for valid coordinates
    if pd.isna(loc1.latitude) or pd.isna(loc2.latitude):
        return None

    # Haversine formula
    R = 6371.0  # Earth radius in km

    lat1, lon1 = radians(loc1.latitude), radians(loc1.longitude)
    lat2, lon2 = radians(loc2.latitude), radians(loc2.longitude)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return round(distance, 2)

def summarize_user_identity_and_needs(entrepreneur_info, model):
    """
    Summarizes user identity and needs based on form data.
    
    Args:
        entrepreneur_info: Dictionary with entrepreneur details
        model: LangChain ChatOpenAI model
        
    Returns:
        str: Summary paragraph describing user identity and needs.
    """
    
    prompt = f"""
    Summarize the following user identity and needs into a concise paragraph:
    
    Entrepreneur overview: {entrepreneur_info}
    
    Use natural language to describe the user basis reported input. Stick to user provided input, don't assume or use any other information source./
    
    The description should reflect user's reported information accurately. Also, specify the "category" in the summary explicitly. For example,
    
    say in terms of "Identity" and profile the user is XXX. In terms of "Growth Stage" the user is XXX. In terms of "Vertical", the user is in XYZ.
    
    In terms of Services, the user needs XYZ.
    """
    
    try:
        # Implement retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.invoke([HumanMessage(content=prompt)])
                return response.content
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                st.warning(f"Retry {attempt+1}/{max_retries} after error: {e}")
    except Exception as e:
        st.error(f"Failed to generate summary: {e}")
        return "Unable to generate summary at this time."


def match_programs_to_entrepreneur(entrepreneur_info, programs_info, model):
    """
    Matches programs to an entrepreneur based on their needs and preferences.
    
    Args:
        entrepreneur_info: Dictionary with entrepreneur details
        programs_info: List of dictionaries with program information
        model: LangChain ChatOpenAI model
        
    Returns:
        List of recommended programs
    """
    prompt = f"""
        You are a smart assistant helping match entrepreneurs to support programs.

        Here is the entrepreneur information:
        {entrepreneur_info}

        Here is the list of available programs and providers:
        {programs_info}

        TASK:
        - Analyze the entrepreneur's needs.
        - Evaluate each program's suitability based on the entrepreneur's needs.
        - For each combination, provide:
            - Entrepreneur ID
            - Provider ID
            - Program Name
            - Need Satisfied
            - Distance score (1-10) where 1 is too far (more than one 1 hour in car or +50 km), 5 is moderate distance (5-10 km drive) and 10 is very close (walking distance less than 10 minutes or <1.5 km).
            - Identity/Profile/Product type/ Growth score (1-10) where 1 is not aligned and 10 is perfectly aligned with the profile, product type and growth stage.
            - Service/Vertical score (1-10) where 1 is not aligned and 5 is perfectly aligned.
            - Need Satisfaction score (1-10) where 1 is not satisfied and 10 is perfectly satisfied.
            - A brief explanation in natural language of why this program is a good match for the entrepreneur's needs based on the above scores.
            - Format the output in **valid JSON**, each containing the above fields.
        - EXAMPLE:

        [
        {{"entrepreneur_d": "E1", "provider_id": "P2", "program_name": "Funding Boost", "need_satisfied": "Funding", "distance_score": 8, "identity_score": 9, "service_score": 10, "need_satisfaction_score": 9, "explanation": "This program provides funding specifically for startups in the tech sector for entrepreneurs in the area X."}},
        {{"entrepreneur_id": "E1", "provider_id": "P3", "program_name": "Mentorship Program", "need_satisfied": "Mentorship", "distance_score": 7, "identity_score": 8, "service_score": 9, "need_satisfaction_score": 8, "explanation": "This program offers mentorship for entrepreneurs in the early growth stage. The program is well-suited for entrepreneurs in the area Y."}},
        ]

        All needs should be evaluated and included in the output even if they are not satisfied by any program.
        Return ONLY valid JSON without any markdown formatting or additional text.
        DO NOT use triple backticks (```) or any other formatting.

    """
    
    try:
        # Implement retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.invoke([HumanMessage(content=prompt)])
                return response.content
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                st.warning(f"Retry {attempt+1}/{max_retries} after error: {e}")
    except Exception as e:
        st.error(f"Failed to match programs: {e}")
        return "Unable to match programs at this time."

def summarize_recommendations(entrepreneur_info, recommendations_response, model):
    """
    Summarizes program recommendations using an LLM.
    
    Args:
        entrepreneur_info: Dictionary with entrepreneur details
        recommendations_response: List of dictionaries with program information
        model: LangChain ChatOpenAI model
        
    Returns:
        Summary string
    """
    # Preserve original prompt
    prompt = f"""
    Summarize the following recommendations into a concise paragraph. Focus on the programs and services that are most relevant to the user based on their final_socre:
    
    Entrepreneur overview: {entrepreneur_info}
    
    Recommendations: {recommendations_response}
    
    Use natural language to describe the recommendations. Stick to user provided input, don't assume or use any other information source.
    """
    
    try:
        # Implement retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.invoke([HumanMessage(content=prompt)])
                return response.content
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                st.warning(f"Retry {attempt+1}/{max_retries} after error: {e}")
    except Exception as e:
        st.error(f"Failed to generate recommendations summary: {e}")
        return "Unable to generate recommendations summary at this time."

def extract_unique_items(df, column_name):
    """
    Extracts unique items from a comma-separated column in a DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The column name containing verticals (default is 'verticals').
        
    Returns:
        list: A sorted list of unique items.
    """
    if df.empty or column_name not in df.columns:
        return []
        
    # Drop NaN values and split the verticals by comma
    lst_split = [x.split(",") for x in df[column_name].dropna().unique().tolist()]
    
    # Flatten the list
    lst_flat = [item for sublist in lst_split for item in sublist]
    
    # Remove duplicates, strip whitespace, and sort the list
    return sorted(list(set([x.strip() for x in lst_flat])))
