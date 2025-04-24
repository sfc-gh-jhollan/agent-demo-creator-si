from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


load_dotenv()

# Initialize OpenAI API
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)


# Define the state schema
class AppState(TypedDict):
    demo_scenario: str
    human_feedback: str
    approved: bool
    final_response: str
    question: str
    script: str
    stack_trace: str
    csv_file: str
    snowflake_stage: str
    semantic_model_yaml: str
    generation: str
    demo_description: str
    question_1: str
    question_2: str
    question_3: str
    question_4: str
    question_5: str
    generated_idea: str
    table_info: list  # Added to store table information
    cortex_search_path: str
    semantic_model_path: str
    schema: str
    agent_name: str
    agent_description_markdown: str
    sample_q_1: str
    sample_q_2: str
    sample_q_3: str
    sample_q_4: str
    sample_q_5: str


# Update the StateGraph to use the defined schema
workflow = StateGraph(AppState)

from nodes.generate_demo_scenario import generate_demo_scenario
from nodes.display_demo_idea import display_demo_idea
from nodes.ask_user_feedback import ask_user_feedback
from nodes.generate_dataset_script import generate_dataset_script
from nodes.generate_document_data import generate_document_data
from nodes.evaluate_human_feedback import evaluate_human_feedback
from nodes.execute_dataset_script import execute_dataset_script
from nodes.upload_to_snowflake import (
    upload_to_snowflake,
    upload_semantic_model,
    create_agent,
    create_cortex_search,
)
from nodes.generate_semantic_model import generate_semantic_model
from nodes.fix_python_script import fix_python_script
from nodes.check_dataset_script import check_dataset_script
from nodes.check_semantic_model import check_semantic_model
from nodes.display_results import display_results
from nodes.generate_agent_description import generate_agent_description

workflow.add_node("GenerateDemoScenario", generate_demo_scenario)
workflow.add_node("DisplayDemoIdea", display_demo_idea)
workflow.add_node("AskUserFeedback", ask_user_feedback)
workflow.add_node("GenerateDocumentData", generate_document_data)
workflow.add_node("GenerateDatasetScript", generate_dataset_script)
workflow.add_node("ExecuteDatasetScript", execute_dataset_script)
workflow.add_node("UploadToSnowflake", upload_to_snowflake)
workflow.add_node("GenerateSemanticModel", generate_semantic_model)
workflow.add_node("FixPythonScript", fix_python_script)
workflow.add_node("UploadSemanticModel", upload_semantic_model)
workflow.add_node("CreateCortexSearch", create_cortex_search)
workflow.add_node("CreateAgent", create_agent)
workflow.add_node("CheckDatasetScript", check_dataset_script)
workflow.add_node("CheckSemanticModel", check_semantic_model)
workflow.add_node("DisplayResults", display_results)
workflow.add_node("GenerateAgentDescription", generate_agent_description)


workflow.add_edge(START, "GenerateDemoScenario")
# Define the flow between nodes
workflow.add_edge("GenerateDemoScenario", "DisplayDemoIdea")
workflow.add_edge("DisplayDemoIdea", "AskUserFeedback")
workflow.add_conditional_edges("AskUserFeedback", evaluate_human_feedback)
workflow.add_edge("GenerateDatasetScript", "CheckDatasetScript")
workflow.add_edge("CheckDatasetScript", "ExecuteDatasetScript")
workflow.add_conditional_edges(
    "ExecuteDatasetScript",
    lambda context, writer: (
        "GenerateDocumentData"
        if not context.get("stack_trace", None)
        else "FixPythonScript"
    ),
)
workflow.add_edge("FixPythonScript", "ExecuteDatasetScript")
workflow.add_edge("GenerateDocumentData", "UploadToSnowflake")
workflow.add_edge("UploadToSnowflake", "GenerateSemanticModel")
workflow.add_edge("GenerateSemanticModel", "CheckSemanticModel")
workflow.add_edge("CheckSemanticModel", "UploadSemanticModel")
workflow.add_edge("UploadSemanticModel", "CreateCortexSearch")
workflow.add_edge("CreateCortexSearch", "GenerateAgentDescription")
workflow.add_edge("GenerateAgentDescription", "CreateAgent")
workflow.add_edge("CreateAgent", "DisplayResults")
workflow.add_edge("DisplayResults", END)


# A checkpointer is required for `interrupt` to work.
checkpointer = MemorySaver()
# Compile the graph after defining nodes and edges
app = workflow.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    app.run()
