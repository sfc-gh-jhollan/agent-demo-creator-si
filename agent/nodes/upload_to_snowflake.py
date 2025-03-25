import os
import pandas as pd
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col


def get_snowflake_session(context):
    # Reuse the session from the context or create a new one if not available
    session = context.get("snowflake_session")
    if not session:
        session = Session.builder.config("CONNECTION_NAME", "si-prod3").getOrCreate()
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

        session.write_pandas(
            df,
            table_name,
            auto_create_table=True,
            overwrite=True,
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


def create_agent(context, writer):
    agent_name = context.get("agent_name", "Demo Agent")
    writer("Creating Cortex Agent...")
    session = get_snowflake_session(context)
    database = session.get_current_database()
    schema = session.get_current_schema()
    session.sql(
        """
        DELETE FROM snowflake_intelligence.agents.config
        WHERE agent_name like '%Demo Agent%';
        """
    ).collect()
    session.sql(
        """
        INSERT into snowflake_intelligence.agents.config
        select
        '{agent_name}',  -- agent_name
        null,          -- agent_description
        ['PUBLIC'],    -- grantee_roles
        -- Tools following the tool_specifications 
        array_construct(
            object_construct(
                    'tool_spec', object_construct(

                'name', 'Documents',
                'type', 'cortex_search'
                )
            ),
            object_construct(
                'tool_spec', object_construct(

                'name', 'Snowflake',
                'type', 'cortex_analyst_text_to_sql'
                )
            )
        ),
        -- Tool resources following the tool_resources documentation. 
        object_construct(
            'Documents', object_construct(
                'id_column', 'DOCUMENT_URL',
                'name', '{cortex_search_path}'
            ),
            'Snowflake', object_construct(
                'semantic_model_file', '{semantic_model_path}'
            )
        ),   -- tool_resources
        null,          -- tool_choice
        NULL;       -- tool_choice_reason
        """.format(
            agent_name=agent_name,
            cortex_search_path=context.get("cortex_search_path"),
            semantic_model_path=context.get("semantic_model_path"),
        )
    ).collect()

    return context
