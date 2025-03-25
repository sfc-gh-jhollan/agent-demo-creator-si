from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


def display_demo_idea(context, writer):
    writer("Processing demo idea...")

    prompt = ChatPromptTemplate.from_template(
        """
        Given the following demo idea for an AI agent demo and questions, inform the user of the demo idea and the potential questions. 

        For the sample questions list them in a list of questions they could copy and paste into the demo agent.

        Be sure to include the annotations, if included, for if SQL or Search for each question.

        Ask for any changes or if they are ok with using this demo to generate datasets and materials.

        ### DEMO IDEA ###
        {demo_description}

        ### QUESTIONS ###
        {questions}
        """,
    )

    llm = ChatOpenAI(model_name="gpt-4o")

    feedback_chain = prompt | llm | StrOutputParser()

    demo_description = context["demo_description"]
    questions = "\n".join(
        [f"Question {i}: {context[f'question_{i}']}" for i in range(1, 6)]
    )

    response = feedback_chain.invoke(
        {"demo_description": demo_description, "questions": questions}
    )

    context["generated_idea"] = response
    return context
