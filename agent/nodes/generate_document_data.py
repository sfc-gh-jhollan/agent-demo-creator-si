from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


def generate_document_data(context, writer):
    writer("Generating document data...")

    class DocumentMetadata(BaseModel):
        title: str = Field(..., description="The title of the document")
        url: str = Field(..., description="The URL of the document")
        generation_description: str = Field(
            ..., description="Description of how the document was generated"
        )

    class DocumentStore(BaseModel):
        documents: list[DocumentMetadata] = Field(
            ..., description="A list of document metadata"
        )

    prompt = ChatPromptTemplate.from_template(
        """
        You are responsible for coming up with documents needed to power the RAG-style document answering AI in a demo for Snowflake. Given the demo description and questions below
        I will be breaking this demo up into two parts. The first is questions that will be answered via SQL / queryable data in Snowflake. You can ignore those questions. The other
        questions will be answered via document retrieval such as RAG. For that, I need to generate some fake documents that will be used to answer the questions.

        Given the demo description below, return a set of document metdata that will be used to answer the questions. DO NOT actually generate the documents. Simply the 
        metadata which includes a prompt that will be passed in to generate a full document in a future step.

        Ideally no more than 5 documents are needed to answer all questions. One document is fine if it is sufficient to answer all RAG related questions.

        DO NOT generate any documents with content that would answer the SQL based questions.
        Only generate documents and document ideas for documents needed to answer the RAG-style questions.

        Return your answer in the following format:
        - documents: A list of document metadata that will be used to generate documents needed to answer the questions.

        ## DEMO DESCRIPTION AND QUESTIONS ##
        - Demo Description: {demo_description}
        - Question 1: {question_1}
        - Question 2: {question_2}
        - Question 3: {question_3}
        - Question 4: {question_4}
        - Question 5: {question_5}
        """,
    )

    llm = ChatOpenAI(model_name="gpt-4o")
    structured_llm_generator = llm.with_structured_output(DocumentStore)

    chain = prompt | structured_llm_generator

    response = chain.invoke(
        {
            "demo_description": context.get("demo_description", ""),
            "question_1": context.get("question_1", ""),
            "question_2": context.get("question_2", ""),
            "question_3": context.get("question_3", ""),
            "question_4": context.get("question_4", ""),
            "question_5": context.get("question_5", ""),
        }
    )
    # Loop through each document in the response
    generated_documents = []
    for document in response.documents:
        writer(f"Generating document for title: {document.title}")

        # Create a prompt to generate the document text
        document_prompt = ChatPromptTemplate.from_template(
            """
            Generate a synthetic document that will be used for a RAG demo based on the following details:

            ## DEMO OVERVIEW ##
            - Demo Description: {demo_description}
            
            ## DOCUMENT DETAILS ##
            Title: {title}
            Instructions for generation: {generation_description}
            """
        )

        chain = document_prompt | llm | StrOutputParser()
        # Use the LLM to generate the document text
        document_text = chain.invoke(
            {
                "demo_description": context.get("demo_description", ""),
                "title": document.title,
                "generation_description": document.generation_description,
            }
        )

        # Append the generated document details to the list
        generated_documents.append(
            {
                "DOCUMENT_TITLE": document.title,
                "DOCUMENT_URL": document.url,
                "TEXT": document_text,
            }
        )

    # Create a CSV file with the generated documents
    import csv

    csv_file_path = "./generated_csvs/DOCUMENTS.csv"
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(
            csv_file, fieldnames=["DOCUMENT_TITLE", "DOCUMENT_URL", "TEXT"]
        )
        csv_writer.writeheader()
        csv_writer.writerows(generated_documents)
