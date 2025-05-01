# app.py
import functions as fn
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
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
import re
from datetime import datetime
from supabase import create_client

# Load environment variables
load_dotenv()
engine = fn.connect_db()

##############################################
# MAP WITH SERVICE PROVIDERS AND ENTREPRENEURS
##############################################

# Streamlit Layout
# innovation icon

st.title("SWPA Innovation Ecosystem")

# Create tabs
about, overview, needs_section, programs, matching = st.tabs(["About", "Rolodex Overview", "Needs", "Programs", "Matching tool"])

with about:
    st.subheader("About this app")
    st.write(
        """
        This app provides an overview of the entrepreneurial ecosystem in Southwestern Pennsylvania (SWPA). 
        It maps the regional service providers and entrepreneurs, helping to evaluate the resource demands and capabilities of local stakeholders.
        Using this app, you can explore the following features:
        - **📍 Service Providers Map**: Visualize the locations of service providers and entrepreneurs in SWPA.
        - **📊 Entrepreneur Needs Distribution**: Analyze the distribution of entrepreneur needs by county, helping to identify areas of demand.
        - **🧩 Program Matching Tool**: Find suitable programs for entrepreneurs based on their needs and the services offered by providers.
        - **📄 Program Details**: Access detailed information about various programs available in the region.
        """
        )
    
    st.markdown("""---""")
    
    st.write("🚀 If you are and entrepreneur looking for support, please fill out the [Entrepreneur Intake Form](https://forms.gle/eMw5PY9QeTXDqPhy6) to help us understand your needs.")

    st.write("🧰 If you are a service provider looking to be included, please fill out the [Service Provider Intake Form](https://forms.gle/aae3SA6YJaZ7d1et5) to help us understand your services.")

with overview:

    st.subheader("📍 Service Providers in Southwestern Pennsylvania")

    query_zipcode_providers = """
    SELECT provider_id as id, provider_name as name, address, zipcode, 'Provider' as user FROM providers
    UNION
    SELECT entrepreneur_id as id, business_name as name, address, zipcode, 'Entrepreneur' as user FROM entrepreneurs
    ;"""

    # query_zipcode_providers = """
    # SELECT provider_id as id, provider_name as name, zipcode, 'Provider' as user FROM providers
    # ;"""
    # @st.cache_data
    # @st.cache_data
    df_zipcode_providers = fn.get_data(query_zipcode_providers, engine)
    # @st.cache_data
    df_zipcode_providers = fn.add_coordinates(df_zipcode_providers)
    map_df = df_zipcode_providers.dropna(subset=['latitude', 'longitude'])

    # Assign color to each row. Red for providers, blue for entrepreneurs
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

    # Legend for the user colors on the map. 
    # The legend should be below the subheader and above the map
    st.markdown(
    """
    <style>
        .legend {
            display: flex;
            justify-content: center;
            margin-bottom: 10px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin-right: 20px;
        }
        .legend-color-box {
            width: 20px;
            height: 20px;
            margin-right: 5px;
            border-radius: 50%; /* Change square to circle */
        }
    </style>
    <div class="legend">
        <div class="legend-item">
            <div class="legend-color-box" style="background-color: #00FF00;"></div>
            <span>Provider</span>
        </div>
        <div class="legend-item">
            <div class="legend-color-box" style="background-color: #0000FF;"></div>
            <span>Entrepreneur</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True
    )


    # st.map(map_df[['latitude', 'longitude']])
    # Scatter with Tooltips ---
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position='[longitude, latitude]',
        get_radius=800,
        get_color="color",
        pickable=True
    )

    # Create county layer
    county_layer = pdk.Layer(
        "GeoJsonLayer",
        data=swpa_counties,
        stroked=True,
        filled=False,
        get_line_color=[0, 0, 150, 255],
        get_line_width=100,
        pickable=False
    )

    view_state = pdk.ViewState(
        latitude=40.4406,
        longitude=-79.9959,
        zoom=8,
        pitch=0
    )

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=view_state,
        layers=[county_layer, layer],
        tooltip={"text": "{user}" + "\n" + "{name}" + "\n" + "{address}" },
    ))


with needs_section:
    ##############################################
    # ENTREPRENEUR NEEDS
    ##############################################

    st.subheader("📌 Distribution of Entrepreneur Needs by County")

    # Query needs data
    query_needs = """
    SELECT en.entrepreneur_id, e.county, en.need, en.service
    FROM entrepreneur_needs AS en
    JOIN entrepreneurs AS e
    ON en.entrepreneur_id = e.entrepreneur_id
    AND en.date_intake = e.date_intake;
    """
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

    # Plot
    fig = px.bar(
        needs_count,
        x='county',
        y='count',
        color='need',
        title="Entrepreneur Needs by County (Filtered by Services)",
        labels={'count': 'Number of Needs', 'county': 'County'},
    )

    fig.update_layout(
        barmode='stack',
        xaxis_title="County",
        yaxis_title="Number of Needs",
        legend_title="Need Type",
        title_x=0.5,
    )

    st.plotly_chart(fig)

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

    fig = px.treemap(
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

    fig.update_traces(
        textfont=dict(
            family='Arial',
            color='black'
        ),
        texttemplate='%{label}'
    )

    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25)
    )

    st.plotly_chart(fig)

with programs:
    ##############################################
    # PROGRAM DETAILS
    ##############################################
    st.subheader("🔍 Programs")
    # @st.cache_data
    df_programs = fn.get_data("""SELECT  DISTINCT ON (t2.provider_id, t2.program_id)
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
                ORDER BY t2.provider_id, t2.program_id, t2.date_intake_form DESC;""", engine)

    # Prepare filter options
    provider_names = sorted(df_programs['provider_name'].dropna().unique().tolist())
    counties = sorted(df_programs['county'].dropna().unique().tolist())
    product_types = fn.extract_unique_items(df_programs, 'product_type')
    verticals = fn.extract_unique_items(df_programs, 'verticals')

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


with matching:
    ##############################################
    # MATCHING ENTREPRENEURS TO PROVIDERS
    ##############################################
    openai_model = ChatOpenAI(
        model="gpt-4o",  # Specify model version (e.g., gpt-4 or gpt-3.5-turbo)
        temperature=0,
        max_tokens=2000,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

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

    # Load programs and providers JSON
    programs_providers = json.dumps(json_data_prov_prog)

    # Your entrepreneurs dataframe (already loaded elsewhere in your app)
    # Assume df_entrep_needs is already loaded

    # Add new section (new tab for assistant)
    st.subheader("🎯 Entrepreneur Assistant: Program Recommendation")

    # Select Entrepreneur
    entrepreneur_business = df_entrep['business_name'].unique().tolist()
    selected_entrepreneur_business = st.selectbox("Select an Entrepreneur", entrepreneur_business)
    # Filter entrepreneur info
    entrepreneur_info = [item for item in json_data_entrep_needs if item.get("business_name") == selected_entrepreneur_business]
    # # Run Button
    run_button = st.button("Run Program Recommendation Assistant")

    if run_button:
            
        # Prepare prompt
        prompt_template = PromptTemplate(
            input_variables=["entrepreneur_info", "programs_info"],
            template="""
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
            - Grwowth Stage score (1-10) where 1 is poorly suited and 10 is perfectly suited.
            - Identity/Profile/Product type score (1-10) where 1 is not aligned and 10 is perfectly aligned.
            - Service/Vertical score (1-10) where 1 is not aligned and 5 is perfectly aligned.
            - Need Satisfaction score (1-10) where 1 is not satisfied and 10 is perfectly satisfied.
            - A brief explanation of why this program is a good match for the entrepreneur's needs.
            - Format the output in **valid JSON**, each containing the above fields.
        - EXAMPLE:

        [
        {{"entrepreneur_d": "E1", "provider_id": "P2", "program_name": "Funding Boost", "need_satisfied": "Funding", "growth_score": 8, "identity_score": 9, "service_score": 10, "need_satisfaction_score": 9, "explanation": "This program provides funding specifically for startups in the tech sector for entrepreneurs in the area X."}},
        {{"entrepreneur_id": "E1", "provider_id": "P3", "program_name": "Mentorship Program", "need_satisfied": "Mentorship", "growth_score": 7, "identity_score": 8, "service_score": 9, "need_satisfaction_score": 8, "explanation": "This program offers mentorship for entrepreneurs in the early growth stage. The program is well-suited for entrepreneurs in the area Y."}},
        ]

        All needs should be evaluated and included in the output even if they are not satisfied by any program.
        Return ONLY valid JSON without any markdown formatting or additional text.
        DO NOT use triple backticks (```) or any other formatting.

        """
        )

        chain = LLMChain(llm=openai_model, prompt=prompt_template)

        # Run LLM
        # response = chain.run(
        #     entrepreneur_info=entrepreneur_info,
        #     programs_info=programs_providers
        # )
        response = chain.invoke({
                                "entrepreneur_info": entrepreneur_info,
                                "programs_info": programs_providers
                            })["text"]

        # Display Entrepreneur Info
        st.subheader("Entrepreneur summary")
        entrepreneur_summary = fn.summarize_user_identity_and_needs(entrepreneur_info, openai_model)
        st.write(entrepreneur_summary)
        # print(response)

        # Try to Parse into DataFrame
        try:
            matches = json.loads(response)
            # final score
            for match in matches:
                match['final_score'] = (match['growth_score'] + match['identity_score'] + match['service_score'] + match['need_satisfaction_score']) / 4
            
            # Display recommendation summary
            st.subheader("Program Recommendations")
            recommendation_summary = fn.summarize_recommendations(entrepreneur_info, matches, openai_model)
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



    ##############################################
    # CHAT ASSISTANT
    ##############################################    

    # st.subheader("🤖 Chat Assistant for Entrepreneur Support")

    # # Session state setup
    # if "chat_history" not in st.session_state:
    #     st.session_state.chat_history = []
    # if "chat_log_df" not in st.session_state:
    #     st.session_state.chat_log_df = pd.DataFrame(columns=[
    #         "timestamp", "entrepreneur_id", "question", "answer"
    #     ])

    # # Load data
    # programs_info = json_data_prov_prog
    # context = {
    #     "entrepreneur_info": entrepreneur_info,
    #     "programs_info": programs_info
    # }
    # context_str = json.dumps(context)

    # # Initialize LLM and memory
    # llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    # memory = ConversationBufferMemory(return_messages=True)

    # # Prompt template
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", "You are an assistant that helps match entrepreneurs with programs based on their needs."),
    #     ("system", "Context: {context}"),
    #     MessagesPlaceholder(variable_name="chat_history"),
    #     ("human", "{input}")
    # ])

    # # Build chain
    # chat_chain = LLMChain(
    #     llm=openai_model,
    #     prompt=prompt,
    #     memory=memory
    # )

    # # Chat input field
    # user_input = st.chat_input("Ask a question about this entrepreneur or the available programs...")

    # if user_input:
    #     # Limit chat to 10 messages (5 user + 5 assistant)
    #     if len(st.session_state.chat_history) >= 10:
    #         st.warning("🔄 Chat history limit reached (10 turns). Clearing memory to start fresh.")
    #         st.session_state.chat_history.clear()
    #         memory.clear()

    #     # Run the assistant
    #     response = chat_chain.run({
    #         "context": context_str,
    #         "input": user_input
    #     })

    #     # Timestamp
    #     now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    #     # Update chat history (display)
    #     st.session_state.chat_history.append({"role": "user", "content": user_input})
    #     st.session_state.chat_history.append({"role": "assistant", "content": response})

    #     # Update log with timestamp
    #     st.session_state.chat_log_df.loc[len(st.session_state.chat_log_df)] = {
    #         "timestamp": now,
    #         "entrepreneur_id": selected_entrepreneur_business,
    #         "question": user_input,
    #         "answer": response
    #     }

    # # Display chat messages
    # for msg in st.session_state.chat_history:
    #     role = "user" if msg["role"] == "user" else "assistant"
    #     st.chat_message(role).write(msg["content"])

    # # Show and download log
    # with st.expander("📄 Show Chat Log Table"):
    #     st.dataframe(st.session_state.chat_log_df)

    # st.download_button(
    #     label="Download Log as CSV",
    #     data=st.session_state.chat_log_df.to_csv(index=False).encode(),
    #     file_name=f"chat_log_{selected_entrepreneur_business}.csv",
    #     mime="text/csv"
    # )