from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# Define conditional nodes to determine the starting point based on the state
def evaluate_human_feedback(context, writer):
    writer("Deciding what to do next...")
    human_feedback = context["human_feedback"]

    # Define the structured output schema
    class FeedbackEvaluationOutput(BaseModel):
        approved: bool = Field(..., description="Whether the demo is approved.")

    prompt = ChatPromptTemplate.from_template(
        """
        Evaluate the user response here. This is a response to a question for feedback on an idea for a demo. 
        Look at what they are saying and determine if any adjustments or changes are being suggested. 
        
        If they approve of the demo based on human feedback, return "true" for the property "approved."

        Return a structured response with the following format:
        - approved: true/false

        ###
        User Response:
        {human_feedback}
        """,
    )

    # LLM with structured output
    llm = ChatOpenAI(model_name="gpt-4o")
    structured_llm_evaluator = llm.with_structured_output(FeedbackEvaluationOutput)

    # Chain
    chain = prompt | structured_llm_evaluator

    # Generate evaluation
    response = chain.invoke(input={"human_feedback": human_feedback})

    # Update the context with evaluation results
    context.update(response.dict())

    # Return the next node based on approval status
    return "GenerateDatasetScript" if response.approved else "GenerateDemoScenario"
