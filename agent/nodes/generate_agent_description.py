from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.types import Command


def generate_agent_description(context, writer):
    writer("Generating Agent description...")

    class AgentDescriptionOutput(BaseModel):
        agent_description_markdown: str = Field(
            ...,
            description="Description of the agent to display in the UI in markdown format.",
        )
        sample_q_1: str = Field(..., description="First sample question.")
        sample_q_2: str = Field(..., description="Second sample question.")
        sample_q_3: str = Field(..., description="Third sample question.")
        sample_q_4: str = Field(..., description="Fourth sample question.")
        sample_q_5: str = Field(..., description="Fifth sample question.")

    prompt_template = """
        You just finished generating everything needed for the user to run a demo in Snowflake Intelligence. As part of it you created a "Data Agent" that has access
        to various data sources which a customer can use to get questions answered. 

        Use the below information to generate the following:
        - Agent Description: A description of the agent that will be displayed in the Snowflake Intelligence UI when users choose this agent. It should be in markdown format, and not in the
        1st person perspective. For example, "This agent can help answer any questions related to customer support flows at Company X. It is connected to data related to ...."
        - Sample Questions: take the question from the demo idea and return a version that could be added as a suggested question. This should largely be the question as is, but if it says (SQL) or (RAG) or (Search), strip those components.

        ## Demo description ##
        {demo_description}

        ## Questions the agent should be able to answer ##
    `   - Question 1: {question_1}
        - Question 2: {question_2}
        - Question 3: {question_3}
        - Question 4: {question_4}
        - Question 5: {question_5}

        ## The semantic model used for Structured / SQL based questions ##
        {semantic_model_yaml}

        """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    llm = ChatOpenAI(model_name="gpt-4o")
    structured_llm_generator = llm.with_structured_output(AgentDescriptionOutput)

    chain = prompt | structured_llm_generator

    with open("semantic_model.yaml", "r") as f:
        semantic_model_yaml = f.read()

    response = chain.invoke(
        {
            "demo_description": context.get("demo_description", ""),
            "question_1": context.get("question_1", ""),
            "question_2": context.get("question_2", ""),
            "question_3": context.get("question_3", ""),
            "question_4": context.get("question_4", ""),
            "question_5": context.get("question_5", ""),
            "semantic_model_yaml": semantic_model_yaml,
        }
    )
    context.update(response.dict())
    return context
