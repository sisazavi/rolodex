# app.py

import streamlit as st
import pandas as pd
import pgeocode
import requests
import json
import os
from dotenv import load_dotenv
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import functions as fn

# Load environment variables
load_dotenv()

# Initialize database connection
@st.cache_resource
def get_db_engine():
    return fn.connect_db()

# Initialize LLM model
@st.cache_resource
def get_llm_model():
    return ChatOpenAI(model="gpt-4o",
                      temperature=0.2,
                      max_tokens=2000,  
                      openai_api_key=fn.get_secret("OPENAI_API_KEY"))

# UI Components
def render_about_tab():
    """Renders the About tab content"""
    st.subheader("About this app")
    st.write(
        """
        This app provides an overview of the entrepreneurial ecosystem in Southwestern Pennsylvania (SWPA). 
        It maps the regional service providers and entrepreneurs, helping to evaluate the resource demands 
        and capabilities of local stakeholders. Using this app, you can explore the following features:
        
        - **📍 Service Providers Map**: Visualize the locations of service providers and entrepreneurs in SWPA.
        - **📊 Entrepreneur Needs Distribution**: Analyze the distribution of entrepreneur needs by county, helping to identify areas of demand.
        - **🧩 Program Matching Tool**: Find suitable programs for entrepreneurs based on their needs and the services offered by providers.
        - **📄 Program Details**: Access detailed information about various programs available in the region.
        """
    )
    st.markdown("""---""")
    st.write("🚀 If you are and entrepreneur looking for support, please fill out the [Entrepreneur Intake Form](https://forms.gle/eMw5PY9QeTXDqPhy6) to help us understand your needs.")
    st.write("🧰 If you are a service provider looking to be included, please fill out the [Service Provider Intake Form](https://forms.gle/aae3SA6YJaZ7d1et5) to help us understand your services.")

def render_overview_tab(engine):
    """Renders the Rolodex Overview tab with map visualization"""
    st.subheader("📍 Service Providers in Southwestern Pennsylvania")
    
    # Original query preserved
    query_zipcode_providers = """
    SELECT provider_id as id, provider_name as name, address, zipcode, 'Provider' as user 
    FROM providers 
    UNION 
    SELECT entrepreneur_id as id, business_name as name, address, zipcode, 'Entrepreneur' as user 
    FROM entrepreneurs;
    """
    
    # Get data and add coordinates
    df_zipcode_providers = fn.get_data(query_zipcode_providers, engine)
    df_zipcode_providers = fn.add_coordinates(df_zipcode_providers)
    map_df = df_zipcode_providers.dropna(subset=['latitude', 'longitude'])
    
    # Assign color to each row
    dynamic_color_map = {
        'Provider': fn.hex_to_rgb('#00FF00'),  # Green
        'Entrepreneur': fn.hex_to_rgb('#0000FF')  # Blue
    }
    map_df['color'] = map_df['user'].map(dynamic_color_map)
    
    # Load and filter counties
    geojson_url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    response = requests.get(geojson_url)
    us_counties = response.json()
    swpa_fips = ["42003", "42005", "42007", "42019", "42051", "42059", "42063", "42073", "42125", "42129"]
    swpa_counties = {
        "type": "FeatureCollection",
        "features": [feature for feature in us_counties['features'] if feature['id'] in swpa_fips]
    }
    
    st.subheader("Service Providers Map")
    
    # Legend for the map
    st.markdown(
        """
        <div style='display: flex; align-items: center; margin-bottom: 10px;'>
            <div style='background-color: #00FF00; width: 15px; height: 15px; margin-right: 5px;'></div>
            <span style='margin-right: 15px;'>Service Provider</span>
            <div style='background-color: #0000FF; width: 15px; height: 15px; margin-right: 5px;'></div>
            <span>Entrepreneur</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Create the map visualization
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["longitude", "latitude"],
        get_color="color",
        get_radius=1000,
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
        radius_min_pixels=5,
        radius_max_pixels=15,
    )
    
    geojson_layer = pdk.Layer(
        "GeoJsonLayer",
        data=swpa_counties,
        opacity=0.2,
        stroked=True,
        filled=True,
        extruded=False,
        wireframe=True,
        get_fill_color=[200, 200, 200],
        get_line_color=[0, 0, 0],
        get_line_width=2,
        line_width_min_pixels=1,
    )
    
    view_state = pdk.ViewState(
        latitude=map_df["latitude"].mean(),
        longitude=map_df["longitude"].mean(),
        zoom=8,
        pitch=0,
    )
    
    map_view = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=view_state,
        layers=[geojson_layer, scatter_layer],
        tooltip={
            "html": "<b>{name}</b><br/>{user}<br/>{address}",
            "style": {"backgroundColor": "white", "color": "black"},
        },
    )
    
    st.pydeck_chart(map_view)
    
    # Display statistics
    provider_count = len(map_df[map_df['user'] == 'Provider'])
    entrepreneur_count = len(map_df[map_df['user'] == 'Entrepreneur'])
    
    st.markdown(f"""
    ### Ecosystem Statistics
    - **Total Service Providers**: {provider_count}
    - **Total Entrepreneurs**: {entrepreneur_count}
    """)

def render_needs_tab(engine):

    ##############################################
    # ENTREPRENEUR NEEDS
    ##############################################
    """Renders the Needs tab with needs analysis"""
    st.subheader("📊 Entrepreneur Needs Analysis")
    
    # Get needs data
    query_needs = """
    SELECT en.entrepreneur_id, e.county, en.need, en.service
    FROM entrepreneur_needs AS en
    JOIN entrepreneurs AS e
    ON en.entrepreneur_id = e.entrepreneur_id
    AND en.date_intake = e.date_intake;
    """
    needs_data = fn.get_data(query_needs, engine)
    
    if not needs_data.empty:
        # @st.cache_data
        needs_df = fn.get_data(query_needs, engine)

        # Multiselect for services
        unique_services = needs_df['service'].dropna().unique().tolist()
        selected_services = st.multiselect(
            "Select Services to Display:",
            options=sorted(unique_services),
            default=sorted(unique_services)
        )

        # Filter needs
        if selected_services:
            filtered_needs_df = needs_df[needs_df['service'].isin(selected_services)]
        else:
            filtered_needs_df = needs_df.copy()

        # Group by county and need
        needs_count = filtered_needs_df.groupby(['county', 'need']).size().reset_index(name='count')

        county_chart = px.bar(
            needs_count,
            x='county',
            y='count',
            color='need',
            title="Entrepreneur Needs by County (Filtered by Services)",
            labels={'count': 'Number of Needs', 'county': 'County'},
        )
        
        county_chart.update_layout(
            barmode='stack',
            xaxis_title="County",
            yaxis_title="Number of Needs",
            legend_title="Need Type",
            title_x=0.5,
        )
        
        st.plotly_chart(county_chart)

        ##############################################
        # ENTREPRENEUR SERVICES NEEDED BY COUNTY
        ##############################################

        # Group needs by County and Service
        needs_grouped = needs_df.groupby(['county', 'service']).size().reset_index(name='count')

        # Calculate total needs per county
        county_totals = needs_grouped.groupby('county')['count'].sum().reset_index(name='total_count')
        needs_grouped = needs_grouped.merge(county_totals, on='county')
        needs_grouped['percent'] = (needs_grouped['count'] / needs_grouped['total_count'] * 100).round(1)

        # Plot Treemap
        st.subheader("Entrepreneur Service type requirement by County")

        treemap = px.treemap(
            needs_grouped,
            path=['county', 'service'],
            values='count',
            color='county',
            color_discrete_sequence=px.colors.qualitative.Plotly,
            hover_data={
                'count': True,
                'percent': True,
                'county': False,
                'service': False,
                'total_count': False
            }
        )

        treemap.update_traces(
            textfont=dict(
                family='Arial',
                color='black'
            ),
            texttemplate='%{label}'
        )

        treemap.update_layout(
            margin=dict(t=50, l=25, r=25, b=25)
        )

        st.plotly_chart(treemap)
    else:
        st.info("No needs data available.")

def render_programs_tab(engine):
    """Renders the Programs tab with program information"""
    st.subheader("🔍📄 Available Programs")
    
    # Get programs data
    query_programs = """SELECT  DISTINCT ON (t2.provider_id, t2.program_id)
                t2.provider_id, t2.provider_name, t2.program_id, t2.program_name,  t2.website,  t2.contact_name,  t2.contact_email, 
                t1.county, t1.address, t2.services, t2.verticals,  t2.product_type,
                CONCAT_WS(' - ', CONCAT('ALL: ',t2.core_audience_all),
                CONCAT('Ecosystem Org (IA, Nonprofit etc): ',t2.core_audience_ecosystem),
                CONCAT('Entrepreneur: ',t2.core_audience_entrepreneur),
                CONCAT('Startups: ',t2.core_audience_startups),
                CONCAT('SMEs/Companies: ',t2.core_audience_sme),
                CONCAT('University Students: ',t2.core_audience_ustudents),
                CONCAT('K-12 Students: ',t2.core_audience_k12students)) as core_audience,
                CONCAT_WS(' - ', 
                    CASE WHEN t2.growth_stage_discovery = 1 THEN 'Discovery/ Idea/ Individual Stage: Poorly suited'
                    WHEN t2.growth_stage_discovery = 2 THEN 'Discovery/ Idea/ Individual Stage: Somewhat suited'
                    WHEN t2.growth_stage_discovery = 3 THEN 'Discovery/ Idea/ Individual Stage: Moderately suited'
                    WHEN t2.growth_stage_discovery = 4 THEN 'Discovery/ Idea/ Individual Stage: Well suited'
                    WHEN t2.growth_stage_discovery = 5 THEN 'Discovery/ Idea/ Individual Stage: Perfectly suited'
                    ELSE 'Discovery/ Idea/ Individual Stage: Not suited' END,
                    CASE WHEN t2.growth_stage_early = 1 THEN 'Early Stage: Poorly suited'
                    WHEN t2.growth_stage_early = 2 THEN 'Early Stage: Somewhat suited'
                    WHEN t2.growth_stage_early = 3 THEN 'Early Stage: Moderately suited'
                    WHEN t2.growth_stage_early = 4 THEN 'Early Stage: Well suited'
                    WHEN t2.growth_stage_early = 5 THEN 'Early Stage: Perfectly suited'
                    ELSE 'Early Stage: Not suited' END,
                    CASE WHEN t2.growth_stage_growth = 1 THEN 'Growth Stage: Poorly suited'
                    WHEN t2.growth_stage_growth = 2 THEN 'Growth Stage: Somewhat suited'
                    WHEN t2.growth_stage_growth = 3 THEN 'Growth Stage: Moderately suited'
                    WHEN t2.growth_stage_growth = 4 THEN 'Growth Stage: Well suited'
                    WHEN t2.growth_stage_growth = 5 THEN 'Growth Stage: Perfectly suited'
                    ELSE 'Growth Stage: Not suited' END,
                    CASE WHEN t2.growth_stage_mature = 1 THEN 'Mature Stage: Poorly suited'
                    WHEN t2.growth_stage_mature = 2 THEN 'Mature Stage: Somewhat suited'
                    WHEN t2.growth_stage_mature = 3 THEN 'Mature Stage: Moderately suited'
                    WHEN t2.growth_stage_mature = 4 THEN 'Mature Stage: Well suited'
                    WHEN t2.growth_stage_mature = 5 THEN 'Mature Stage: Perfectly suited'
                    ELSE 'Mature Stage: Not suited' END
                ) AS growth_stage
                FROM  programs t2 INNER JOIN providers t1
                    ON t1.provider_id = t2.provider_id
                ORDER BY t2.provider_id, t2.program_id, t2.date_intake_form DESC;"""
    
    df_programs = fn.get_data(query_programs, engine)
    
    if not df_programs.empty:
    
    ##############################################
    # PROGRAMS OVERVIEW BY VERTICALS AND COUNTIES
    ##############################################
        
        # Create a bar chart for verticals and counties
        st.subheader("Programs Overview by Verticals and Counties")
        
        # Multiselect for verticals
        verticals = fn.extract_unique_items(df_programs, 'verticals')
        if '' in verticals:
            verticals.remove('')

        selected_verticals = st.multiselect(
            "Select Verticals to Display:",
            options=sorted(verticals),
            default=sorted(verticals)
        )
        
        #counties
        counties = sorted(df_programs['county'].dropna().unique().tolist())

        # Filter programs by selected verticals
        if selected_verticals:
            df_programs_filtered = df_programs[df_programs['verticals'].isin(selected_verticals)]
        else:
            df_programs_filtered = df_programs.copy()
        
        
        # Reorganize the data for better visualization. For each vertical in verticals, create a new row for each county.
        verticals_count2 = {'verticals': [], 'county': [], 'count': []}
        for vert in verticals:
            for row in df_programs[['county', 'verticals']].itertuples():
                # print(vert, "is in:", row.verticals,": ", vert in row.verticals)
                if vert in row.verticals:
                    verticals_count2['verticals'].append(vert)
                    verticals_count2['county'].append(row.county)
                    verticals_count2['count'].append(1)

        verticals_count = pd.DataFrame(verticals_count2)

        # Group by verticals and counties
        verticals_count = verticals_count.groupby(['verticals', 'county']).sum('count').reset_index()

        # Create a bar chart
        program_verticals = px.bar(
            verticals_count,
            x='county',
            y='count',
            color='verticals',
            title="Programs Overview by Verticals and Counties",
            labels={'count': 'Number of Programs', 'county': 'County'},
        )
        
        program_verticals.update_layout(
            barmode='stack',
            xaxis_title="County",
            yaxis_title="Number of Programs",
            legend_title="Vertical",
            title_x=0.5,
        )
        
        st.plotly_chart(program_verticals)
    
    ##############################################
    # PROGRAM DETAILS
    ##############################################
    
    # Prepare filter options
        provider_names = sorted(df_programs['provider_name'].dropna().unique().tolist())
        product_types = fn.extract_unique_items(df_programs, 'product_type')

        # Build filters
        col1, col2 = st.columns(2)
        with col1:
            selected_provider = st.selectbox("Filter by Provider", ["All"] + provider_names)
            selected_county = st.selectbox("Filter by County", ["All"] + counties)
        with col2:
            selected_product_type = st.selectbox("Filter by Product Type", ["All"] + product_types)
            selected_vertical = st.selectbox("Filter by Vertical", ["All"] + verticals)

        # Apply filters
        filtered_programs = df_programs.copy()

        if selected_provider != "All":
            filtered_programs = filtered_programs[filtered_programs['provider_name'] == selected_provider]
        if selected_county != "All":
            filtered_programs = filtered_programs[filtered_programs['county'] == selected_county]
        if selected_product_type != "All":
            filtered_programs = filtered_programs[filtered_programs['product_type'].str.contains(selected_product_type, na=False)]
        if selected_vertical != "All":
            filtered_programs = filtered_programs[filtered_programs['verticals'].str.contains(selected_vertical, na=False)]

        # Display
        st.markdown(f"**{len(filtered_programs)}** program(s) match the selected criteria.")
        st.dataframe(filtered_programs)
    else:
        st.info("No programs data available.")

def render_matching_tab(engine, model):
    """Renders the Matching tool tab"""
    st.subheader("🎯 Entrepreneur Assistant: Program Recommendation")
    
    ##############################################
    # MATCHING ENTREPRENEURS TO PROVIDERS
    ##############################################

    query_providers = """SELECT DISTINCT ON (provider_id)
                                provider_id,
                                provider_name,
                                address,
                                description,
                                zipcode,
                                county,
                                "BBB",
                                programs_available
                                FROM providers
                                ORDER BY provider_id, date_intake_form DESC; """
    
    query_programs = """
    SELECT  DISTINCT ON (provider_id, program_id)
    provider_id, program_id,
    program_name,  website,  contact_name,  contact_email,  services,
    CONCAT_WS(' - ', CONCAT('ALL: ',core_audience_all),
    CONCAT('Ecosystem Org (IA, Nonprofit etc): ',core_audience_ecosystem),
    CONCAT('Entrepreneur: ',core_audience_entrepreneur),
    CONCAT('Startups: ',core_audience_startups),
    CONCAT('SMEs/Companies: ',core_audience_sme),
    CONCAT('University Students: ',core_audience_ustudents),
    CONCAT('K-12 Students: ',core_audience_k12students)) as core_audience,
    CONCAT_WS(' - ', 
        CASE WHEN growth_stage_discovery = 1 THEN 'Discovery/ Idea/ Individual Stage: Poorly suited'
        WHEN growth_stage_discovery = 2 THEN 'Discovery/ Idea/ Individual Stage: Somewhat suited'
        WHEN growth_stage_discovery = 3 THEN 'Discovery/ Idea/ Individual Stage: Moderately suited'
        WHEN growth_stage_discovery = 4 THEN 'Discovery/ Idea/ Individual Stage: Well suited'
        WHEN growth_stage_discovery = 5 THEN 'Discovery/ Idea/ Individual Stage: Perfectly suited'
        ELSE 'Discovery/ Idea/ Individual Stage: Not suited' END,
        CASE WHEN growth_stage_early = 1 THEN 'Early Stage: Poorly suited'
        WHEN growth_stage_early = 2 THEN 'Early Stage: Somewhat suited'
        WHEN growth_stage_early = 3 THEN 'Early Stage: Moderately suited'
        WHEN growth_stage_early = 4 THEN 'Early Stage: Well suited'
        WHEN growth_stage_early = 5 THEN 'Early Stage: Perfectly suited'
        ELSE 'Early Stage: Not suited' END,
        CASE WHEN growth_stage_growth = 1 THEN 'Growth Stage: Poorly suited'
        WHEN growth_stage_growth = 2 THEN 'Growth Stage: Somewhat suited'
        WHEN growth_stage_growth = 3 THEN 'Growth Stage: Moderately suited'
        WHEN growth_stage_growth = 4 THEN 'Growth Stage: Well suited'
        WHEN growth_stage_growth = 5 THEN 'Growth Stage: Perfectly suited'
        ELSE 'Growth Stage: Not suited' END,
        CASE WHEN growth_stage_mature = 1 THEN 'Mature Stage: Poorly suited'
        WHEN growth_stage_mature = 2 THEN 'Mature Stage: Somewhat suited'
        WHEN growth_stage_mature = 3 THEN 'Mature Stage: Moderately suited'
        WHEN growth_stage_mature = 4 THEN 'Mature Stage: Well suited'
        WHEN growth_stage_mature = 5 THEN 'Mature Stage: Perfectly suited'
        ELSE 'Mature Stage: Not suited' END
    ) AS growth_stage,
    verticals,  product_type,  scraped_description
    FROM programs
    ORDER BY provider_id, program_id, date_intake_form DESC;"""
    # @st.cache_data
    df_providers = fn.get_data(query_providers, engine)
    # @st.cache_data
    df_programs = fn.get_data(query_programs, engine)

    # Convert the DataFrame to JSON format
    json_data_prov_prog = fn.df_to_json_nest(df_providers, df_programs, join_key="provider_id", child_key="programs")

    query_entrepreneurs = """
    SELECT  DISTINCT ON (entrepreneur_id) entrepreneur_id, name,  business_name,  
    email,  phone,  address,  zipcode,  website,  profile,
    growth_stage,  vertical,  county
    FROM entrepreneurs
    ORDER BY entrepreneur_id, date_intake DESC;
    ;"""

    query_needs = """
    SELECT DISTINCT ON (entrepreneur_id, need, date_intake) 
    entrepreneur_id, service,  need
    FROM entrepreneur_needs
    ORDER BY entrepreneur_id, need, date_intake DESC"""
    # @st.cache_data
    df_entrep = fn.get_data(query_entrepreneurs, engine)
    # @st.cache_data
    df_needs = fn.get_data(query_needs, engine)

    # Convert the DataFrame to JSON format
    json_data_entrep_needs = fn.df_to_json_nest(df_entrep, df_needs, join_key="entrepreneur_id", child_key="needs_needed")

    # Your entrepreneurs dataframe (already loaded elsewhere in your app)
    # Assume df_entrep_needs is already loaded

    # Select Entrepreneur
    entrepreneur_business = df_entrep['business_name'].unique().tolist()
    selected_entrepreneur_business = st.selectbox("Select an Entrepreneur", entrepreneur_business)
    # Filter entrepreneur info
    entrepreneur_info = [item for item in json_data_entrep_needs if item.get("business_name") == selected_entrepreneur_business]

    # estimate the distance between the entrepreneur and the providers using fn.estimate_zipcode_distance
    if entrepreneur_info:
        entrepreneur_info = entrepreneur_info[0]
        entrepreneur_zipcode = entrepreneur_info.get("zipcode")
        if entrepreneur_zipcode:
            for provider in json_data_prov_prog:
                provider_zipcode = provider['zipcode']
            # Calculate distances and add to JSON data
                provider['distance'] = fn.estimate_zipcode_distance(
                    entrepreneur_zipcode, provider_zipcode)
    
    # Load programs and providers JSON
    programs_providers = json.dumps(json_data_prov_prog)

    # # Run Button
    run_button = st.button("Run Program Recommendation Assistant")

    if run_button:
            
        # Match Entrepreneur to Providers
        program_recomendation_response = fn.match_programs_to_entrepreneur(entrepreneur_info, programs_providers, model)

        # Display Entrepreneur Info
        st.subheader("Entrepreneur summary")
        entrepreneur_summary = fn.summarize_user_identity_and_needs(entrepreneur_info, model)
        st.write(entrepreneur_summary)

        # Try to Parse into DataFrame
        try:
            matches = json.loads(program_recomendation_response)
            # final score
            for match in matches:
                match['final_score'] = (match['distance_score'] + match['identity_score'] + match['service_score'] + match['need_satisfaction_score']) / 4
            
            # Display recommendation summary
            st.subheader("Program Recommendations")
            recommendation_summary = fn.summarize_recommendations(entrepreneur_info, matches, model)
            st.write(recommendation_summary)

            matches_df = pd.DataFrame(matches).sort_values(by='final_score', ascending=False)
            
            st.subheader("Structured Recommendations")
            st.dataframe(matches_df)
            
            # Button to insert into database
            if st.button("Send Results to Database"):
                matches_df["date"] = pd.Timestamp.now()
                fn.insert_data_to_supabase(matches_df, 'needs_match')

        except Exception as e:
            st.error(f"Failed to parse assistant response into structured data: {e}")


def main():
    """Main application entry point"""
    st.title("SWPA Innovation Ecosystem")
    
    # Initialize resources
    engine = get_db_engine()
    model = get_llm_model()
    
    # Create tabs
    about, overview, needs_section, programs, matching = st.tabs([
        "About", "Rolodex Overview", "Needs", "Programs", "Matching tool"
    ])
    
    # Render each tab
    with about:
        render_about_tab()
    
    with overview:
        render_overview_tab(engine)
    
    with needs_section:
        render_needs_tab(engine)
    
    with programs:
        render_programs_tab(engine)
    
    with matching:
        render_matching_tab(engine, model)

if __name__ == "__main__":
    main()
