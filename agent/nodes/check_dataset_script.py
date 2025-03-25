from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from datetime import datetime


def check_dataset_script(context, writer):
    writer("Checking the synthetic generation script...")

    class DemoScript(BaseModel):
        script: str = (
            Field(
                ...,
                description="Python script contents (.py) file to generate synthetic datasets.",
            ),
        )

    current_date = datetime.now().strftime("%B %d, %Y")

    prompt = ChatPromptTemplate.from_template(
        """
        We are creating a demo for Snowflake and AI. The following instructions were given to an LLM to generate a Python script
        to create synthetic data to support the demo. Can you check the script and return the script either exactly as-is, or with
        any adjustments or modifications to improve it.

        DO NOT include any content besides what would be in a python script file.

        Return the results in the following format:
        - script: Python script contents (.py) file to generate synthetic datasets. 

        ## INSTRUCTIONS GIVEN FOR INITIAL GENERATION ##
        This is a new product that uses AI Agents to help organizations understand what is happening in their business around data.
        It has a few pieces of functionality - 1) ability to take a question from a user and generate / execute SQL based on data that
        is in their Snowflake account. This can include joins as needed as well. 2) ability to take a question from a user and route to a
        RAG-style answer path where appropriate context is surfaced from business documents (for instance, Sharepoint, Google Drive, Slack, Confluence,
            Support Tickets, etc.) to answer a question with citations. 3) It can also allow users to just interact with an LLM and do things like
            upload file and talk to documents.
        
        Here is the demo description and the questions that are intended to be answered. For the python script you generate, only generate for the questions
        that will be answered via SQL execution (not the RAG-style document ones).

        You can assume the python environment that will execute this has `pandas`, `numpy`, `random`, and `faker` available. Ensure the generated script can be copied into a .py file and would run, so include all needed import statements.

        Write the script so that it writes the synthetic data to a CSV file, with the format TABLENAME.csv in the current directory. So for example, if the table should be called CUSTOMERS it writes CUSTOMERS.csv with the generated data.

        Save all CSVs into the folder `generated_csvs` in the current directory.

        Joins are supported, but make sure your script generates valid IDs so after I load the synthetic data into snowflake the SQL queries would actually work.

        Confirm that the synthetic data tables created have all of the columns and datapoints needed so the suggested question could be answered by a single SQL query.

        Be aware of the current date ({current_date}) when generating data, and ensure that any date-related data aligns with this context.

        Ensure the script is indented correctly and is syntactically correct as it will be executed in the next step.

        Confirm the data generated can answer all structured SQL questions with all necessary columns required to answer.

        ### DEMO DESCRIPTION ###
        {demo_description}

        ### QUESTIONS THAT NEED ANSWERING ###
        Pay attention for generating data only for those that require SQL to answer.
        - Question 1: {question_1}
        - Question 2: {question_2}
        - Question 3: {question_3}
        - Question 4: {question_4}
        - Question 5: {question_5}

        ### THINGS TO EVALUATE AND LOOK FOR ###
        - Ensure the script generates valid IDs for joins.
        - Ensure that any join has identical column names for the join.
        - Ensure the script is indented correctly.
        - Ensure the script is syntactically correct.
        - Ensure the script generates all necessary columns required to answer the questions. If any columns are missing, please adjust the script to generate.
        - Ensure the script generates enough data to answer the questions.

        ### CURRENT SCRIPT ###
        ```python
        {script}
        ```
        """,
    )

    llm = ChatOpenAI(model_name="gpt-4o")
    structured_llm_generator = llm.with_structured_output(DemoScript)

    chain = prompt | structured_llm_generator

    response = chain.invoke(
        {
            "demo_description": context.get("demo_description", ""),
            "question_1": context.get("question_1", ""),
            "question_2": context.get("question_2", ""),
            "question_3": context.get("question_3", ""),
            "question_4": context.get("question_4", ""),
            "question_5": context.get("question_5", ""),
            "current_date": current_date,
            "script": context.get("script", ""),
        }
    )

    with open("generated_script.py", "w") as f:
        f.write(response.script)
    context.update(response.dict())
    return context
