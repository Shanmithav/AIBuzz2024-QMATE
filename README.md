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


Some Samples Questions that you can ask based on the dummy email inputs given

**Project Alpha General Information**
"What is the current status of Project Alpha?"
"Who is leading Project Alpha?"
"What teams are involved in Project Alpha?"
"What is the timeline for Project Alpha?"
"When is Project Alpha scheduled to launch?"
**QGenie Integration**
"What is the QGenie Implementation Project?"
"Why was API-based integration chosen over local model deployment?"
"What are the phases of QGenie implementation?"
"When will the QGenie Embedding Service be completed?"
"What performance benefits does QGenie's cloud endpoints provide?"
"What are the main API endpoints for QGenie?"
"What rate limits are implemented for the QGenie API?"
"Who is leading the QGenie implementation?"
**Budget & Resources**
"What budget changes were requested for Project Alpha?"
"Why was a budget increase needed for Project Alpha?"
"How much additional budget was requested?"
"Who needs to approve the budget for Project Alpha?"
"When will the new team members be onboarded?"
"Why is the team being expanded?"
**Marketing & Business Strategy**
"What is the marketing strategy for Project Alpha?"
"Why is the marketing focusing on digital channels and influencers?"
"When does the marketing campaign for Project Alpha start?"
"What is the expected ROI for the marketing approach?"
"What partnership opportunity has been proposed for Project Alpha?"
"What benefits could the TechGiant partnership bring?"
"When does TechGiant want to announce the partnership?"
"What are the success criteria for Project Alpha?"
**Compliance & Legal**
"What compliance issues were identified for Project Alpha?"
"What changes are needed to comply with GDPR and CCPA?"
"What are the potential penalties for non-compliance?"
"Who is responsible for ensuring compliance?"
"When must the compliance changes be implemented?"
**Testing & Quality Assurance**
"What is the testing strategy for Project Alpha?"
"What types of testing will be performed on Project Alpha?"
"Who is leading the testing efforts?"
"What testing environments are being used?"
"Why was continuous testing implemented in the CI/CD pipeline?"
"When will the testing framework be set up?"
**People & Teams**
"Who is Sarah Lee and what is her role?"
"Who is John Doe and what is his role?"
"Who is David Chen and what is his role?"
"Who is Alex Chen and what is his role?"
"Who is Maria Garcia and what is her role?"
"Who is Emily Rodriguez and what is her role?"
"What is the relationship between Sarah Lee and John Doe?"
"Which teams are working on Project Alpha?"
**Timeline & Milestones**
"What are the key milestones for Project Alpha?"
"When will the database migration be completed?"
"When will the UI changes be implemented and tested?"
"What is happening in January 2024 for Project Alpha?"
"What deadlines are coming up in December 2023?"
"What is the timeline for the marketing campaign?"
"When will the hiring process begin for new team members?"
**Cross-Document Analysis**
"What connections exist between the QGenie implementation and Project Alpha?"
"How do the technical architecture decisions impact the marketing strategy?"
"What is the relationship between the user testing results and the budget increase?"
"How does the TechGiant partnership affect the team expansion plans?"
"What impact do the compliance requirements have on the technical architecture?"
"How do the testing strategies align with the project timeline?"
"What are all the decisions that have been made across all projects?




