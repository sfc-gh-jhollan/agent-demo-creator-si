# Agent Demo Creator

An Agent of Agents

## Setup

1. Clone this repository, and navigate to the folder
2. Create a virtual environment

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3. Install the requirements

    ```bash
    pip install -r requirements.txt
    ```

5. Create a database in your Snowflake account the agent can create schemas / datasets in. For example:

    ```sql
    CREATE DATABASE SNOWFLAKE_INTELLIGENCE_DEMOS;
    ```

4. Make sure you have Snowflake configuration file with a connection set for it to connect to. This should be in `~/.snowflake/config.toml`. The role will need to have permission to create tables, cortex search services, and add config to `snowflake_intelligence.agents.config`. For example:

    ```toml
    [connections.agent-creator]
    account = "<your-account>"
    user = "<your-username>"
    authenticator = "externalbrowser"
    database = "SNOWFLAKE_INTELLIGENCE_DEMOS"
    warehouse = "SNOWFLAKE_INTELLIGENCE_WH"
    role = "ACCOUNTADMIN"
    ```

5. Rename `.env.sample` to `.env` and replace the values with the right values for your environment.

6. Run the Streamlit UI

    ```bash
    streamlit run streamlit/app.py
    ```
