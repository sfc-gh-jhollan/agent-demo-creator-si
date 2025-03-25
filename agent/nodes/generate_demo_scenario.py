from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.types import Command


def generate_demo_scenario(context, writer):
    writer("Generating a potential demo scenario...")

    class DemoScenarioOutput(BaseModel):
        demo_description: str = Field(
            ..., description="Description of the demo scenario."
        )
        question_1: str = Field(..., description="First question related to the demo.")
        question_2: str = Field(..., description="Second question related to the demo.")
        question_3: str = Field(..., description="Third question related to the demo.")
        question_4: str = Field(..., description="Fourth question related to the demo.")
        question_5: str = Field(..., description="Fifth question related to the demo.")

    prompt_template_base = """
            You are a demo idea generator for Snowflake. Specifically help someone come up with a compelling and interesting idea for Snowflake
            Intelligence. This is a new product that uses AI Agents to help organizations understand what is happening in their business around data.
            It has a few pieces of functionality - 1) ability to take a question from a user and generate / execute SQL based on data that
            is in their Snowflake acccount. This can include joins as needed as well. 2) ability to take a question from a user and route to a
            RAG-style answer path where appropriate context is surfaced from business documents (for instance, Sharepoint, Google Drive, Slack, Confluence,
             Support Tickets, etc.) to answer a question with citations. 3) It can also allow users to just interact with an LLM and do things like
              upload file and talk to documents.
            
            When coming up with a demo idea, the target \"end user\" of the Snowflake Intelligence product is a BUSINESS USER. So do not suggest
            something where a customer of a company is interacting. For example, if generating a demo for Disney, the target user would be
            someone in charge of planning, staffing, support, marketing, sales, or other internal functions. It would not be a Disney consumer interacting with this AI.
            Generally we are pitching this demo to the data teams who are in charge of all the data for business functions, so choose a
            scenario that is most likely to resonate with them broadly.

            In general the demo flow will be explaining some scenario, and then walking through a few questions and seeing how Snowflake Intelligence
            can answer. Usually about 2-3 questions should require SQL execution, ideally at least one of them is rendered by the AI as a chart
            like a line chart. Then 1 or 2 questions can require the Search or RAG style answering. 

            After creating the demo idea, we will use AI to help generate a synthetic dataset to load into Snowflake Intelligence.

            For each question, YOU MUST annotate the question with a category if either "SQL" or "Search" (or RAG) to indicate which strategy would be used to answer. 
            Each question can only have one category. Adding a parenthentical before the question is a good way to annotate.
              
            Here is some guidance from the user on the type of demo or company they are considering: {question}.
            Provide the following outputs:
            - demo_description
            - question_1
            - question_2
            - question_3
            - question_4
            - question_5
            """
    if context.get("human_feedback"):
        prompt_final = (
            prompt_template_base
            + """  
            We already suggested this demo idea with the following questions:
            - Demo Description: {demo_description}
            - Question 1: {question_1}
            - Question 2: {question_2}
            - Question 3: {question_3}
            - Question 4: {question_4}
            - Question 5: {question_5}
            
            They provided this feedback: {human_feedback}.
            """
        )

    else:
        prompt_final = prompt_template_base

    prompt = ChatPromptTemplate.from_template(prompt_final)

    llm = ChatOpenAI(model_name="gpt-4o")
    structured_llm_generator = llm.with_structured_output(DemoScenarioOutput)

    chain = prompt | structured_llm_generator

    response = chain.invoke(
        {
            "question": context["question"],
            "demo_description": context.get("demo_description", ""),
            "question_1": context.get("question_1", ""),
            "question_2": context.get("question_2", ""),
            "question_3": context.get("question_3", ""),
            "question_4": context.get("question_4", ""),
            "question_5": context.get("question_5", ""),
            "human_feedback": context.get("human_feedback", ""),
        }
    )
    context.update(response.dict())
    return context
