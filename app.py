import os
import re
import pandas as pd
import streamlit as st
from typing import List, Dict, Any
import openai
from pinecone import Pinecone, ServerlessSpec
import json
from openai import OpenAI
from dotenv import load_dotenv
import time
import matplotlib.pyplot as plt

load_dotenv()

client = OpenAI()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Ensure the Pinecone API key is set
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Function to create the index with retry logic
def create_index_with_retry(index_name, dimension, metric, spec, retries=3, delay=5):
    for attempt in range(retries):
        try:
            if index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=spec
                )
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                st.error(f"Failed to create index after {retries} attempts: {e}")
                return False

index_name = "3gpp-index"
dimension = 1536
metric = 'euclidean'
spec = ServerlessSpec(cloud='aws', region='us-east-1')

# Attempt to create the index
index_created = create_index_with_retry(index_name, dimension, metric, spec)

MODEL = "gpt-4o"


class PenTestVAPTAssistant:
    def __init__(self, index_name, embeddings_model="text-embedding-3-small", llm_model=MODEL):
        self.openai = openai
        self.pinecone = pc
        self.index = self.pinecone.Index(index_name)
        self.embeddings_model = embeddings_model
        self.llm_model = llm_model
        self.client = OpenAI()

    def generate_embedding(self, text):
        try:
            response = self.client.embeddings.create(input=text, model=self.embeddings_model)
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Error generating embedding: {e}")
            return None
    
    def search_index(self, query, top_k=6):
        embedding = self.generate_embedding(query)
        if embedding is None:
            return None
        try:
            query_result = self.index.query(
                vector=embedding,
                top_k=top_k,
                include_values=False,
                include_metadata=True
            )
            return query_result
        except Exception as e:
            st.error(f"Error querying index: {e}")
            return None

    def retrieve_documents(self, query_result, max_docs=3):
        documents = []
        if not query_result or 'matches' not in query_result:
            return documents
        for result in query_result['matches'][:max_docs]:
            try:
                document_text = result['metadata']['content']
                documents.append(document_text)
            except KeyError:
                st.error(f"Document ID '{result['id']}' not found in metadata.")
        return documents

    def generate_report(self, query, documents):
        prompt = f"Question: {query}\n\nRelevant Documents:\n"
        for doc in documents:
            prompt += f"- {doc}\n"
        prompt += "\nProvide a detailed answer to the question above based on the relevant documents. Include references in the format 'Reference: [source information]'."
        
        role_description = (
    "= Your Role =\n"
    "Your primary role is to provide detailed, accurate information about the technical specifications outlined in the provided documents, such as TS 38.101-1. "
    "You must have a comprehensive understanding of the content, including changes between versions, key technical requirements, and operational specifications.\n\n"
    "You are supplied with several documents and datasets to answer the questions. Please go through all the documents exhaustively to find the answer logically. If you need to use web searches or general knowledge, do so and mention that to the user.\n\n"
    "THE SPECIFIC DETAILS OR QUESTIONS MUST BE SUPPLIED BY THE USER. IF THE USER DID NOT SUPPLY THE NECESSARY INFORMATION OR QUESTIONS IN THE CHAT, STOP AND ASK FOR THEM FIRST. RELY ONLY ON THE USER-SUPPLIED INFORMATION AND YOUR BUILT-IN KNOWLEDGE BASE.\n\n"
    "= Your Job =\n"
    "Upon receiving a query about the technical specifications or a section of a report, your job is to analyze the request carefully and provide clear, concise, and compelling answers that highlight the relevant information from the document. "
    "Your responses should be tailored to address the specific needs or concerns indicated in the query.\n\n"
    "You MUST not hallucinate on technical data; you must consult your knowledge base. The user is going to make important decisions based on this information, so always please provide references to internal knowledge or external sources for the information you are providing.\n\n"
    "= Instructions on Actions =\n"
    "You should leverage your extensive knowledge base about technical specifications, including detailed descriptions, version comparisons, and key requirements. While you may not have external browsing capabilities, your built-in knowledge should be comprehensive and up-to-date.\n\n"
    "= Your knowledge base =\n"
    "- For detailed technical specifications and changes, refer to the provided documents in your knowledge base.\n"
    "- Look for comparisons between versions in the change logs provided in the document.\n\n"
    "= Guidelines =\n"
    "- For understanding specific technical terms or standards, refer to the relevant sections of the document.\n\n"
    "= How to Work with User Queries =\n"
    "1. Clarify the query or section provided by the user to ensure you fully understand the request and the context.\n"
    "2. Identify key points that need to be addressed in your response, focusing on providing accurate and relevant information from the document.\n"
    "3. Organize your response logically, starting with an introduction, followed by detailed points addressing each aspect of the query, and concluding with a strong summary or call to action.\n"
    "4. Ensure that your responses are specific to the user's request, avoiding generic or overly broad statements.\n"
    "5. You must not hallucinate.\n"
    "6. You must give references to your responses every time. If there is no response in the knowledge base, please say so when you have to go externally and search.\n\n"
    "= Outputs =\n"
    "Your output should include:\n"
    "- Detailed answers to queries about technical specifications, showcasing the relevance and accuracy of the information.\n"
    "- You must present your logic and thought process on how you arrive at the answer, including references.\n"
    "- Clear and factual language that is tailored to the audience and purpose of the query.\n"
    "- Any necessary clarification questions to the user if the provided information is insufficient to craft a complete response.\n"
    "- You must include the references of the documents you are referring to in your response.\n"
    "- You must present your thoughts and lay out any assumptions in detail to the user.\n"
    "- Use tables or summaries when appropriate to present the information succinctly and clearly.\n"
    "- If any new version of the document is referred to, compare and contrast it with the older version."
)



#         role_description = (
#     "### Your Role\n"
#     "Your primary role is to provide detailed, accurate information and analysis of Supreme Court of India judgments based on PDFs provided. "
#     "You must have a comprehensive understanding of the legal terms, court procedures, and the ability to extract and summarize key points from the judgments to effectively communicate this information to potential users, including researchers, legal professionals, and policymakers.\n\n"
#     "You are supplied with several documents and datasets to answer the questions. Please go through all the documents exhaustively to find the answer logically. When you have to use web searches or general knowledge, do so and mention that to the user.\n\n"
#     "THE SPECIFIC DETAILS OR QUESTIONS MUST BE SUPPLIED BY THE USER. IF THE USER DID NOT SUPPLY THE NECESSARY INFORMATION OR QUESTIONS IN THE CHAT, STOP AND ASK FOR THEM FIRST. RELY ONLY ON THE USER-SUPPLIED INFORMATION AND YOUR BUILT-IN KNOWLEDGE BASE.\n\n"
#     "### Your Job\n"
#     "Upon receiving a query about a Supreme Court judgment or a section of a PDF to respond to, your job is to analyze the request carefully and provide clear, concise, and compelling answers that highlight the relevant legal points and outcomes. "
#     "Your responses should be tailored to address the specific needs or concerns indicated in the query.\n\n"
#     "You MUST not hallucinate on legal data; you must consult your knowledge base. The user is going to make important decisions based on this information, so always please provide references to internal knowledge or external sources for the information you are providing.\n\n"
#     "### Instructions on Actions\n"
#     "You should leverage your extensive knowledge base about Supreme Court judgments, including detailed descriptions, legal analyses, case outcomes, and historical comparisons. "
#     "While you may not have external browsing capabilities, your built-in knowledge should be comprehensive and up-to-date.\n\n"
#     "### Your Knowledge Base\n"
#     "- For detailed legal analyses and historical case data, refer to the judgments_data.xlsx in your knowledge base.\n"
#     "- Look for case summaries and key legal points in the judgments_overview.pdf in your knowledge base.\n\n"
#     "### Guidelines\n"
#     "- For legal terminologies and standards, you must look at legal_standards.pdf.\n"
#     "- Refer to the case indexing system to locate specific judgments and related documents efficiently.\n\n"
#     "### How to Work with User Queries\n"
#     "1. **Clarify the Query**: Ensure you fully understand the request and the context. Ask for necessary details if not provided.\n"
#     "2. **Identify Key Points**: Focus on providing accurate and relevant legal data.\n"
#     "3. **Organize Your Response**: Start with an introduction, followed by detailed points addressing each aspect of the query, and conclude with a strong summary.\n"
#     "4. **Specific Responses**: Avoid generic or overly broad statements. Address the user's specific request.\n"
#     "5. **No Hallucinations**: Ensure your responses are based on your knowledge base. Indicate if you need to refer to external sources.\n"
#     "6. **Provide References**: Include references from your knowledge base for every response.\n\n"
#     "### Outputs\n"
#     "Your output should include:\n"
#     "- Detailed answers to queries about Supreme Court judgments, showcasing the relevance and accuracy of the information.\n"
#     "- Your logic and thought process on how you arrive at the answer, including references.\n"
#     "- Clear and factual language tailored to the audience and purpose of the query.\n"
#     "- Any necessary clarification questions to the user if the provided information is insufficient to craft a complete response.\n"
#     "- References of the documents you are referring to in your response.\n"
#     "- Detailed assumptions and thoughts presented to the user.\n"
#     "- Use of tables, charts, and other visualizations for clear and succinct presentation of information.\n\n"
#     "### Important Note\n"
#     "1. Ensure accuracy in legal data and case summaries.\n"
#     "2. Mention referance number of Pdf as well "
#     "3. Maintain confidentiality and sensitivity towards legal information."
# )

        
        
        
        messages = [
            {"role": "system", "content": role_description},
            {"role": "user", "content": prompt}
        ]
        
        completion_params = {
            "model": self.llm_model,
            "messages": messages
        }
        
        try:
            response = self.client.chat.completions.create(**completion_params)
            report = response.choices[0].message.content.strip()
            
            references = self.extract_references(report)
            
            return report, references
        except Exception as e:
            st.error(f"An error occurred while generating the report: {e}")
            return None, []

    def extract_references(self, report):
        lines = report.split("\n")
        references = [line for line in lines if line.startswith("Reference:")]
        return references

    def query(self, query):
        query_result = self.search_index(query)
        if query_result is None:
            return None, []
        documents = self.retrieve_documents(query_result)
        if not documents:
            st.warning("No relevant documents found.")
            return None, []
        report, references = self.generate_report(query, documents)
        return report, references

# Streamlit app layout
st.set_page_config(page_title="3GPP-GPT", layout="wide")

# Custom CSS for stylingststre
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
        }
        .main-title {
            font-size: 2.5rem;
            color: #000000;
            text-align: center;
            margin-bottom: 25px;
        }
        .description {
            font-size: 1.2rem;
            color: #333;
            text-align: center;
            margin-bottom: 50px;
        }
        .sidebar .sidebar-content {
            background-color: #F8F8FF;
        }
        .dataframe {
            margin: 20px;
            border-collapse: collapse;
        }
        .dataframe th {
            background-color: #4B0082;
            color: white;
        }
        .dataframe td, .dataframe th {
            padding: 10px;
            border: 1px solid #ddd;
        }
        .expander-header {
            font-size: 1.5rem;
            color: #4B0082;
        }
        .expander-content {
            font-size: 1.1rem;
            color: #555;
        }
        .expander-content p {
            margin: 10px 0;
        }
        input[type="text"] {
            autocomplete: off;
        }
    </style>
""", unsafe_allow_html=True)




# Title and description
st.markdown("<h1 class='main-title'>3GPP-GPT</h1>", unsafe_allow_html=True)
st.markdown("<p class='description'>This dashboard allows you to ask questions and get answers from the model. Your question history will be displayed on the left-hand side.</p>", unsafe_allow_html=True)


# User interaction - Real-time chatbot
with st.form(key='question_form'):
    user_question = st.text_input("Enter your question here:", autocomplete='off')
    submit_button = st.form_submit_button(label='Ask')

# Initialize session state for storing history
if 'history' not in st.session_state:
    st.session_state.history = [] 

# Create the PenTestVAPTAssistant instance
assistant = PenTestVAPTAssistant(index_name=index_name)

# Placeholder for the answer
answer_placeholder = st.empty()
references_placeholder = st.empty()
chart_placeholder = st.empty()

if submit_button:
    if user_question.strip() != "":
        # Clear the previous answer and chart
        answer_placeholder.empty()
        references_placeholder.empty()
        chart_placeholder.empty()

        report, references = assistant.query(user_question)
        if report:
            st.session_state.history.append({
                "question": user_question,
                "answer": report,
                "references": references
            })
            
            # Display the new answer
            answer_placeholder.markdown(f"**Answer:**")
            lines = report.split("\n")
            print(lines)
            for line in lines:
                if "[\text" in line:  # Check if the line contains LaTeX
                    st.latex(line)
                else:
                    st.markdown(line)
            # Display references if any
            if references:
                references_placeholder.markdown("**References:**")
                for ref in references:
                    references_placeholder.markdown(f"- {ref}")

            # Display a sample chart
            
        else:
            st.write("No response generated.")
    else:
        st.write("Please enter a question.")

# Display question history only if there are questions in the history
if st.session_state.history:
    st.sidebar.write("### Question History")
    for i, entry in enumerate(st.session_state.history):
        if st.sidebar.button(entry['question'], key=f"history_{i}"):
            user_question = entry['question']
            answer_placeholder.empty()
            answer_placeholder.markdown(f"**Answer:** {entry['answer']}")
            references_placeholder.empty()
            if entry['references']:
                references_placeholder.markdown("**References:**")
                for ref in entry['references']:
                    references_placeholder.markdown(f"- {ref}")
            chart_placeholder.empty()

            # Optionally, you can display the chart related to the selected history question
