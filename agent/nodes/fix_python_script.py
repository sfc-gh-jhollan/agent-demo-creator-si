from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def fix_python_script(context, writer):
    writer("Fixing Python script due to an error during execution...")

    class DemoScript(BaseModel):
        script: str = Field(
            ...,
            description="Python script contents (.py) file to generate synthetic datasets.",
        )

    stack_trace = context.get("stack_trace", "")
    current_script = context.get("script", "")

    prompt = ChatPromptTemplate.from_template(
        """
        A Python script was generated to create synthetic datasets for a demo, but it encountered an error during execution. 
        Below is the script and the error stack trace. Please fix the script to address the error and ensure it runs successfully.

        DO NOT include any content besides what would be in a python script file.

        Return the results in the following format:
        - script: Python script contents (.py) file to generate synthetic datasets.


        ### Current Script ###
        {current_script}

        ### Error Stack Trace ###
        {stack_trace}
        """,
    )

    llm = ChatOpenAI(model_name="gpt-4o")
    structured_llm_generator = llm.with_structured_output(DemoScript)

    chain = prompt | structured_llm_generator

    response = chain.invoke(
        {"current_script": current_script, "stack_trace": stack_trace}
    )
    context["stack_trace"] = None

    context.update(response.dict())
    return context
