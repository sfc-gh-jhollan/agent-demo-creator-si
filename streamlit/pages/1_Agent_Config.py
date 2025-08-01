import os
import streamlit as st
from snowflake.snowpark import Session
from dotenv import load_dotenv
import pandas as pd
import json

# Load environment variables
load_dotenv()

# Initialize Snowflake session
session = Session.builder.config(
    "CONNECTION_NAME", os.getenv("SNOWFLAKE_CONNECTION_NAME", "agent-creator")
).getOrCreate()

st.title("Agent Configuration Management")

# Fetch agent configurations
st.header("Manage Agent Configurations")
agents_query = "SHOW AGENTS IN ACCOUNT"
agents_df = session.sql(agents_query).to_pandas()
print(agents_df)

if not agents_df.empty:
    agent_name = st.selectbox(
        "Select an Agent to View Details:", agents_df['"name"'].tolist()
    )

    if agent_name:
        agent_details = agents_df[agents_df['"name"'] == agent_name].iloc[0]
        st.subheader(f"Details for Agent: {agent_name}")
        st.write(agent_details)

        if st.button(f"Delete Agent: {agent_name}"):
            delete_query = f"DROP AGENT SNOWFLAKE_INTELLIGENCE.AGENTS.{agent_name}"
            session.sql(delete_query).collect()
            st.success(f"Agent {agent_name} deleted successfully.")
else:
    st.write("No agents found.")

# Fetch schemas
st.header("Manage Schemas")
schemas_query = "SHOW SCHEMAS"
schemas_result = session.sql(schemas_query).collect()

schemas_df = pd.DataFrame([{"name": row["name"]} for row in schemas_result])

if not schemas_df.empty:
    schema_name = st.selectbox("Select a Schema to Drop:", schemas_df["name"].tolist())

    if schema_name:
        if st.button(f"Drop Schema: {schema_name}"):
            drop_query = f"DROP SCHEMA {schema_name}"
            session.sql(drop_query).collect()
            st.success(f"Schema {schema_name} dropped successfully.")
else:
    st.write("No schemas found.")
