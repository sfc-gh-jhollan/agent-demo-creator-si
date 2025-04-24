from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from datetime import datetime


def check_semantic_model(context, writer):
    writer("Checking the semantic model...")

    class DemoScript(BaseModel):
        semantic_model_yaml: str = Field(
            ...,
            description="The semantic model .yaml file for a Snowflake semantic model",
        )

    prompt = ChatPromptTemplate.from_template(
        """
        You are responsible for checking the work of another LLM. The LLM created a YAML file for a Snowflake semantic model.
        Please check the semantic model for accuracy. Return either the semantic model as-is or a revised version.

        DO NOT include any content besides the semantic model as text.

        ### INSTRUCTIONS GIVEN TO ORIGINAL PROMPT ###
        You are responsible for looking at a set of tables in Snowflake and generating a semantic model for them. This is in service of a demo that will be using
        the underlying data store for tables described below to answer a few questions that are listed below as well. Do note that some of the questions may
        not be answerable via SQL and will need RAG-style documents to answer which will mostly be handled separate from the semantic model.

        YOU MUST include a set of synonyms for each column.
        YOU MUST include a description for each table, especially that will help in answering the questions.
        YOU MUST include a set of sample values for a table.
        YOU MUST ensure that `name`, `description`, `tables`, and `relationships` are all top level YAML properties (not nested under any other property).
        YOU MUST ensure that any relationships have identical columns names for `left_column` and `right_column`. If the column names are not identical, no relationship exists.
        YOU MUST ensure any table defined in a relationship has `primary_key` columns defined in the `tables` section.

        DO NOT include any verified queries or metrics in the semantic model.
        DO NOT include ``` characters in the response
        DO NOT add any properties or aspects to YAML that aren't explicitly documented
        DO NOT create any relationships that do not have a identical column name in both tables. 
          If you see two ID fields in two tables, that's a good indicator a relationship exists.

        Below you will find sections for the following:
        
        TABLE INFO - which contains the information about each table in snowflake including the full path to the table, the columns, and sample values.
        Be aware of synonyms and other hints in the semantic model that will improve accuracy of answering the specific questions listed below.
        
        DEMO DESCRIPTION AND QUESTIONS - this will contain the demo description and the questions that are intended to be answered by the semantic model. As mentioned some of the questions
        may not be answered by SQL and will rather be answered by RAG-style documents.

        SEMANTIC MODEL DOCUMENTATION - documentation from Snowflake on the syntax and example semantic models. Use this to generate a valid YAML file.

        Return the results in the following format:
        - semantic_model_yaml: A yaml file that is a valid Snowflake semantic model based on the data and Snowflake account provided.

        Be sure the string you return is syntantically valid YAML, including consistent indentation and proper quoting of strings.
        
        ### TABLE INFO ###

        {table_info}

        ## DEMO DESCRIPTION AND QUESTIONS ##
        - Demo Description: {demo_description}
        - Question 1: {question_1}
        - Question 2: {question_2}
        - Question 3: {question_3}
        - Question 4: {question_4}
        - Question 5: {question_5}

        ## SEMANTIC MODEL DOCUMENTATION ##

        {semantic_model_documentation}

        ## CURRENT GENERATED SEMANTIC MODEL ##

        ```yaml
        {semantic_model_yaml}
        ```

        ## THINGS TO CONFIRM ##
        - If relationships exist, the left and right columns defined have identical values. If they are not identical or no left and right columns are defined, there's likely no relationship in the synthentic dataset.
        - If a relationship is defined, all tables included in a relationship have one or many `primary_key` defined in the `tables` section.
        - Formatting and indentation is consistent and correct. `name`, `description`, `tables`, and `relationships` are all top level YAML properties, and should not be nested or indented under any other property.
        - Table descriptions and column synonyms are sufficient to capture any context in included questions.

        """,
    )

    llm = ChatOpenAI(model_name="gpt-4o")
    structured_llm_generator = llm.with_structured_output(DemoScript)

    chain = prompt | structured_llm_generator

    with open("semantic_model_docs.txt", "r") as f:
        semantic_model_documentation = f.read()

    response = chain.invoke(
        {
            "table_info": context["table_info"],
            "demo_description": context.get("demo_description", ""),
            "question_1": context.get("question_1", ""),
            "question_2": context.get("question_2", ""),
            "question_3": context.get("question_3", ""),
            "question_4": context.get("question_4", ""),
            "question_5": context.get("question_5", ""),
            "semantic_model_documentation": semantic_model_documentation,
            "semantic_model_yaml": context.get("semantic_model_yaml", ""),
        }
    )
    yaml_content = response.semantic_model_yaml.strip()

    # Remove leading ```yaml or ```
    if yaml_content.startswith("```yaml"):
        yaml_content = yaml_content[7:]
    elif yaml_content.startswith("```"):
        yaml_content = yaml_content[3:]

    # Remove trailing ``` and any text after it
    if "```" in yaml_content:
        yaml_content = yaml_content.split("```", 1)[0]

    yaml_content = yaml_content.strip()

    with open("semantic_model.yaml", "w") as f:
        f.write(yaml_content)
    context.update(response.dict())
    return context
