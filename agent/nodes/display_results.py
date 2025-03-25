from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


def display_results(context, writer):
    writer("Wrapping up...")

    prompt = ChatPromptTemplate.from_template(
        """
        You just finished generating everything needed for the user to run a demo in Snowflake Intelligence.

        Let the user know the details based on the below:

        - The name of the agent now available in Snowflake Intelligence: {agent_name}
        - The schema that was created that has all tables and the semantic model: {schema}
        - The location of the semantic model is in a stage called "MODELS" and a file called "semantic_model.yaml"
        - Here are the questions they can now ask the agent. Ideally display them so they are easy to copy/paste:
        -- question 1: {question_1}
        -- question 2: {question_2}
        -- question 3: {question_3}
        -- question 4: {question_4}
        -- question 5: {question_5}
        """,
    )

    llm = ChatOpenAI(model_name="gpt-4o")

    feedback_chain = prompt | llm | StrOutputParser()

    response = feedback_chain.invoke(
        {
            "agent_name": context.get("agent_name", ""),
            "schema": context.get("schema", ""),
            "question_1": context.get("question_1", ""),
            "question_2": context.get("question_2", ""),
            "question_3": context.get("question_3", ""),
            "question_4": context.get("question_4", ""),
            "question_5": context.get("question_5", ""),
        }
    )
    context["final_response"] = response
    return context
