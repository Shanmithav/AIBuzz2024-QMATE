Team Name: QueryMates

Idea Title: QMATE (Don't worry buddy I got it all)

Idea url: https://aibuzz.qualcomm.com/idea/4899

Team Members: Shanmitha Vangeti, Pruthvi Raj Singh

Programming Language Used:
Python

AI Hub Model Links
QGenie API (https://qgenie-chat.qualcomm.com)
QGenieChat model
QGenieLLM model
Custom QGenieEmbeddings wrapper
Target Device
  PC
Implementation Summary
  Q-MATE is an AI-powered organizational memory assistant that helps users retrieve and visualize information from various document sources. It processes emails and Confluence pages to extract key information, builds a knowledge graph of entities and relationships, and provides a natural language interface for querying organizational knowledge. The system uses     QGenie's AI models for document summarization, information extraction, and question answering, with an interactive visualization of the knowledge graph.

Key features include:
    Natural language querying of organizational knowledge
    AI-generated document summaries
    Extraction of decisions, insights, people, and timelines
    Interactive visualization of entity relationships
    Context-aware answers with source attribution
    Installation & Setup Steps
    Install required Python packages:

Installations: 
  pip install streamlit networkx pyvis nltk torch numpy requests langchain-core
  Install QGenie SDK Basic [ https://qgenie-sdk-python.qualcomm.com/qgenie_sdk_core/tutorials/basic_usage.html ]
  pip install qgenie
  Download NLTK data
There is a place holder in the code to add ypur QGenie API key and endpoint 
Run the application with Streamlit:

Run
streamlit run AI_project_qgenie.py
Expected Output / Behavior
  Web interface with "Q-MATE: Don't Worry Buddy I Got It All" title
  Sidebar with configuration options and "Load/Reload All Data" button
  Two tabs: "AI Agent & Memory Graph" and "Document Summaries"
  In the AI Agent tab:
  Text input for asking questions
  Answer display with contextual information panels
  Interactive knowledge graph visualization
  Source attribution for answers
  In the Document Summaries tab:
  Dropdown to select documents
  AI-generated summaries
  Option to view original document content
  Sidebar showing recent questions and key entities
  Additional Steps Required for Running
  The application requires a valid QGenie API key and access to the QGenie endpoint
  SSL verification is disabled in the code, which may be necessary for corporate environments
  The application uses simulated data (hardcoded emails and Confluence pages) for demonstration purposes
  Sufficient memory is required to handle document embeddings and graph operations
  Host URL: http://localhost:8501/
Note : 
    The code is now using some hardcoded emails and confluence pages , for the aiu models to read and summarize as there are restrictions to handle enterprise emials, once we have access to enterprize EMIALS , we can add the Email API endpoint to get real-time-data.
Output UI Images : 

<img width="2552" height="1309" alt="image" src="https://github.com/user-attachments/assets/fdc36955-4620-4f29-81be-3ec2b9fa8551" />
<img width="2556" height="1316" alt="image" src="https://github.com/user-attachments/assets/2abf14d7-940c-43c9-8999-278878a38573" />



