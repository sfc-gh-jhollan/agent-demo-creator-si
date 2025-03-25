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
agents_query = (
    "SELECT AGENT_NAME, TOOLS, TOOL_RESOURCES FROM SNOWFLAKE_INTELLIGENCE.AGENTS.CONFIG"
)
agents_df = session.sql(agents_query).to_pandas()

if not agents_df.empty:
    agent_name = st.selectbox(
        "Select an Agent to View Details:", agents_df["AGENT_NAME"].tolist()
    )

    if agent_name:
        agent_details = agents_df[agents_df["AGENT_NAME"] == agent_name].iloc[0]
        st.subheader(f"Details for Agent: {agent_name}")
        st.json(json.loads(agent_details["TOOLS"]))
        st.json(json.loads(agent_details["TOOL_RESOURCES"]))

        if st.button(f"Delete Agent: {agent_name}"):
            delete_query = f"DELETE FROM SNOWFLAKE_INTELLIGENCE.AGENTS.CONFIG WHERE AGENT_NAME = '{agent_name}'"
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
