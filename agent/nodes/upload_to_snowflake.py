import os
import pandas as pd
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


def get_snowflake_session(context):
    # Reuse the session from the context or create a new one if not available
    session = context.get("snowflake_session")
    if not session:
        session = Session.builder.config(
            "CONNECTION_NAME", os.getenv("SNOWFLAKE_CONNECTION_NAME", "agent-creator")
        ).getOrCreate()
        schema = context.get("schema", "DEFAULT")
        session.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}").collect()
        session.use_schema(schema)
        context["snowflake_session"] = session
    return session


def upload_to_snowflake(context, writer):
    writer("Creating data in Snowflake (check for MFA notifications)...")

    session = get_snowflake_session(context)

    csv_directory = "./generated_csvs"
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith(".csv")]

    table_info = []

    for csv_file in csv_files:
        table_name = os.path.splitext(csv_file)[0].upper()
        writer(f"Uploading {csv_file} to Snowflake as table {table_name}...")

        file_path = os.path.join(csv_directory, csv_file)
        df = pd.read_csv(file_path)

        df.columns = [col.upper() for col in df.columns]

        # Drop existing table and detect/convert date-like columns
        session.sql(f"DROP TABLE IF EXISTS {table_name}").collect()
        date_cols = set()
        # Only convert columns where at least 80% of values match the YYYY-MM-DD pattern
        date_threshold = 0.8
        for col_name in df.columns:
            series_str = df[col_name].astype(str)
            mask = series_str.str.match(r"^\d{4}-\d{2}-\d{2}$")
            if mask.sum() >= len(df) * date_threshold:
                # safely parse strictly formatted dates
                df[col_name] = pd.to_datetime(
                    df[col_name], format="%Y-%m-%d", errors="coerce"
                ).dt.date
                date_cols.add(col_name)

        # Build column definitions based on detected types
        col_defs = []
        for col_name in df.columns:
            if col_name in date_cols:
                col_type = "DATE"
            elif pd.api.types.is_integer_dtype(df[col_name]):
                col_type = "NUMBER"
            elif pd.api.types.is_float_dtype(df[col_name]):
                col_type = "FLOAT"
            else:
                max_len = df[col_name].astype(str).map(len).max() or 1
                col_type = f"VARCHAR({max_len})"
            col_defs.append(f"{col_name} {col_type}")

        # Create table with schema and load data
        session.sql(f"CREATE TABLE {table_name} ({', '.join(col_defs)})").collect()
        session.write_pandas(
            df,
            table_name,
            auto_create_table=False,
            overwrite=False,
            quote_identifiers=False,
        )

        # Check if the first column has 'ID' in its name (case insensitive)
        first_column = df.columns[0]
        if "ID" in first_column.upper():
            session.sql(
                f"ALTER TABLE {table_name} ADD PRIMARY KEY ({first_column})"
            ).collect()

        snowflake_table = session.table(table_name)
        column_info = []
        for column in snowflake_table.schema.fields:
            column_name = column.name
            column_type = column.datatype
            sample_values = (
                snowflake_table.select(col(column_name))
                .limit(5)
                .to_pandas()[column_name]
                .tolist()
            )
            column_info.append(
                {
                    "column_name": column_name,
                    "column_type": str(column_type),
                    "sample_values": sample_values,
                }
            )

        database = session.get_current_database()
        schema = session.get_current_schema()
        fully_qualified_name = f"{database}.{schema}.{table_name}"

        if table_name != "DOCUMENTS":
            table_info.append(
                {
                    "table_name": table_name,
                    "fully_qualified_name": fully_qualified_name,
                    "columns": str(column_info),  # Convert column_info to a string
                }
            )

    context["snowflake_stage"] = "uploaded_stage"
    context["table_info"] = table_info
    return context


def upload_semantic_model(context, writer):
    writer("Uploading semantic model to Snowflake...")

    session = get_snowflake_session(context)

    database = session.get_current_database().replace('"', "")
    schema = session.get_current_schema().replace('"', "")
    stage_name = "MODELS"
    file_path = "./semantic_model.yaml"

    # Create the stage if it doesn't exist and ensure it has a directory table enabled
    session.sql(
        f"CREATE STAGE IF NOT EXISTS {stage_name} DIRECTORY = (ENABLE = TRUE)"
    ).collect()

    # Upload the file to the stage
    session.file.put(file_path, f"@{stage_name}", overwrite=True, auto_compress=False)

    context["semantic_model_path"] = f"@{database}.{schema}.MODELS/semantic_model.yaml"
    return context


def create_cortex_search(context, writer):
    writer("Creating Cortex Search...")

    session = get_snowflake_session(context)

    database = session.get_current_database().replace('"', "")
    schema = session.get_current_schema().replace('"', "")
    session.sql(
        """
        CREATE OR REPLACE CORTEX SEARCH SERVICE SEARCH 
        ON TEXT 
        ATTRIBUTES
            DOCUMENT_TITLE,DOCUMENT_URL 
        WAREHOUSE = SNOWFLAKE_INTELLIGENCE_WH 
        EMBEDDING_MODEL = 'snowflake-arctic-embed-m-v1.5' 
        TARGET_LAG = '1 day' 
        AS (
            SELECT
                TEXT,DOCUMENT_TITLE,DOCUMENT_URL
            FROM DOCUMENTS
        );
"""
    ).collect()

    context["cortex_search_path"] = f"{database}.{schema}.SEARCH"

    return context


def generate_tool_descriptions(context, writer):
    """Generate descriptions for the tools using LLM based on semantic model and documents"""
    writer("Generating tool descriptions using LLM...")

    # Generate description for Snowflake_Data tool based on semantic model
    semantic_model_prompt = ChatPromptTemplate.from_template(
        """
        Based on the following semantic model YAML content, generate a comprehensive description for a Cortex Analyst tool.
        The description should explain what tables are available, their purpose, key columns, and how they relate to each other.
        Format it as a detailed technical description that would help an agent understand what data it can query.
        
        Include details about:
        - Each table's purpose and contents
        - Key columns and their meanings
        - Relationships between tables
        - The overall reasoning for how these tables work together
        
        Semantic Model:
        {semantic_model_yaml}
        """
    )

    # Generate description for Documents tool based on CSV content
    documents_prompt = ChatPromptTemplate.from_template(
        """
        Based on the document data that will be available in the Cortex Search service, generate a brief description
        of what types of documents and information are available for search.
        
        Document types and content:
        {document_info}
        
        Generate a concise description (1-2 sentences) explaining what documents are available for search.
        """
    )

    llm = ChatOpenAI(model_name="gpt-4o")

    # Read semantic model
    try:
        with open("semantic_model.yaml", "r") as f:
            semantic_model_yaml = f.read()
    except FileNotFoundError:
        semantic_model_yaml = "Semantic model not available"

    # Read documents info
    try:
        with open("./generated_csvs/DOCUMENTS.csv", "r") as f:
            # Read first few lines to understand document types
            lines = f.readlines()[:6]  # Header + 5 sample rows
            document_info = "\n".join(lines)
    except FileNotFoundError:
        document_info = "Document data not available"

    # Generate descriptions
    semantic_chain = semantic_model_prompt | llm | StrOutputParser()
    documents_chain = documents_prompt | llm | StrOutputParser()

    snowflake_data_description = semantic_chain.invoke(
        {"semantic_model_yaml": semantic_model_yaml}
    )

    documents_description = documents_chain.invoke({"document_info": document_info})

    context["snowflake_data_description"] = snowflake_data_description
    context["documents_description"] = documents_description

    return context


def create_agent(context, writer):
    agent_name = context.get("agent_name", "Demo Agent")
    writer("Creating Cortex Agent...")
    session = get_snowflake_session(context)
    database = session.get_current_database()
    schema = session.get_current_schema()

    # Get the warehouse name from session or use default
    try:
        warehouse = session.get_current_warehouse() or "SNOWFLAKE_INTELLIGENCE_WH"
    except:
        warehouse = "SNOWFLAKE_INTELLIGENCE_WH"

    # Create the agent specification JSON
    agent_spec = {
        "models": {"orchestration": "auto"},
        "orchestration": {},
        "instructions": {
            "sample_questions": [
                {"question": context.get("sample_q_1", "")},
                {"question": context.get("sample_q_2", "")},
                {"question": context.get("sample_q_3", "")},
                {"question": context.get("sample_q_4", "")},
                {"question": context.get("sample_q_5", "")},
            ]
        },
        "tools": [
            {
                "tool_spec": {
                    "type": "cortex_analyst_text_to_sql",
                    "name": "Snowflake_Data",
                    "description": context.get(
                        "snowflake_data_description",
                        "Access to structured data via SQL queries",
                    ),
                }
            },
            {
                "tool_spec": {
                    "type": "cortex_search",
                    "name": "Documents",
                    "description": context.get(
                        "documents_description", "Search through document repository"
                    ),
                }
            },
        ],
        "tool_resources": {
            "Documents": {
                "id_column": "DOCUMENT_URL",
                "max_results": 10,
                "name": context.get("cortex_search_path"),
            },
            "Snowflake_Data": {
                "execution_environment": {"type": "warehouse", "warehouse": warehouse},
                "semantic_model_file": context.get("semantic_model_path"),
            },
        },
    }

    import json

    agent_spec_json = json.dumps(agent_spec)

    # Clean descriptions for JSON compatibility
    agent_description = (
        context.get("agent_description_markdown", "")
        .replace('"', '\\"')
        .replace("'", "\\'")
    )

    # Create the agent using parameterized query approach
    create_agent_sql = f"""
        CREATE AGENT SNOWFLAKE_INTELLIGENCE.AGENTS.{agent_name.replace(" ", "_").upper()}
        WITH PROFILE='{{"display_name": "{agent_name}"}}'
        COMMENT=$${agent_description}$$
        FROM SPECIFICATION $${agent_spec_json}$$;
    """

    session.sql(create_agent_sql).collect()

    return context
