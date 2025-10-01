import os
import ssl
import requests
from functools import wraps
import re
import nltk
import streamlit as st
import networkx as nx
import datetime
from pyvis.network import Network

# --- Ensure Hugging Face environment variables for models if still relevant ---
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

# --- Patch requests to always use verify=False ---
def patch_requests():
    original_request = requests.sessions.Session.request
    @wraps(original_request)
    def new_request(self, method, url, **kwargs):
        kwargs['verify'] = False
        return original_request(self, method, url, **kwargs)
    requests.sessions.Session.request = new_request
patch_requests()

# --- Disable SSL verification for certs ---
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# --- QGenie SDK and LangChain Imports ---
import qgenie # For qgenie.util.cos_sim
from qgenie import QGenieClient
# Import specific LangChain wrappers based on your QGenie SDK documentation
from qgenie.integrations.langchain import QGenieChat, QGenieLLM
# Custom wrapper for QGenieLLM to provide embedding functionality
from langchain_core.embeddings import Embeddings
import numpy as np

class QGenieEmbeddings(Embeddings):
    """
    Custom wrapper class for QGenieLLM to provide embedding functionality.
    This implements the LangChain Embeddings interface using QGenieLLM.
    """
    
    def __init__(self, llm_model):
        """Initialize with a QGenieLLM instance."""
        self.llm = llm_model
        
    def embed_documents(self, texts):
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embeddings, one for each text
        """
        try:
            # If the LLM has a direct embed_documents method, use it
            if hasattr(self.llm, 'embed_documents'):
                return self.llm.embed_documents(texts)
                
            # Otherwise, generate embeddings one by one
            # In a production environment, you would implement a proper embedding method
            # that calls the QGenie API's embedding endpoint
            embeddings = []
            for text in texts:
                # For demonstration purposes, we're creating deterministic embeddings
                # based on the hash of the text content
                text_hash = hash(text) % 10000
                np.random.seed(text_hash)
                embeddings.append(np.random.rand(768))  # Typical embedding dimension
                
            return embeddings
            
        except Exception as e:
            logging.error(f"Error in embed_documents: {e}")
            # Fallback to random embeddings
            return [np.random.rand(768) for _ in texts]
            
    def embed_query(self, text):
        """
        Generate an embedding for a query text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding for the text
        """
        try:
            # If the LLM has a direct embed_query method, use it
            if hasattr(self.llm, 'embed_query'):
                return self.llm.embed_query(text)
                
            # Otherwise, generate an embedding
            # In a production environment, you would implement a proper embedding method
            # that calls the QGenie API's embedding endpoint
            
            # For demonstration purposes, we're creating a deterministic embedding
            # based on the hash of the query content
            text_hash = hash(text) % 10000
            np.random.seed(text_hash)
            return np.random.rand(768)  # Typical embedding dimension
            
        except Exception as e:
            logging.error(f"Error in embed_query: {e}")
            # Fallback to random embedding
            return np.random.rand(768)

# From LangChain itself for RAG
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import torch

# --- Configuration ---
GRAPH_HTML_PATH = "graph.html"

# --- NLTK Downloads ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- Logging ---
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global QGenie/LangChain Components ---
QGENIE_CLIENT = None
QGENIE_EMBEDDINGS = None
QGENIE_LLM = None # For summarization, extraction, and answering

@st.cache_resource
def load_qgenie_components():
    logger.info("Attempting to initialize QGenie client and LangChain components...")
    try:
        # Define endpoint and API key once
        QGENIE_ENDPOINT = "https://qgenie-chat.qualcomm.com"
        QGENIE_API_KEY = "8a289064-1224-443b-a2f8-75d75d437ac0" # <--- IMPORTANT: REPLACE WITH YOUR QGENIE API KEY

        # 1. Initialize the QGenie Client
        qgenie_client = QGenieClient(endpoint=QGENIE_ENDPOINT, api_key=QGENIE_API_KEY)
        logger.info("QGenieClient initialized successfully.")

        # 2. Initialize QGenie LLM for embeddings (will be wrapped by QGenieEmbeddings)
        qgenie_llm_for_embeddings = QGenieLLM(
            qgenie_client=qgenie_client,
            api_key=QGENIE_API_KEY,
            # model_id="your-embedding-model-id" # Optional: specify if needed
        )
        
        # Wrap the LLM with our custom QGenieEmbeddings class
        qgenie_embeddings = QGenieEmbeddings(llm_model=qgenie_llm_for_embeddings)
        logger.info("QGenieEmbeddings wrapper initialized successfully.")

        # 3. Initialize QGenie LLM for summarization, extraction, and RAG (LangChain component)
        qgenie_llm = QGenieChat(
            qgenie_client=qgenie_client,
            api_key=QGENIE_API_KEY,
            # model_id="your-llm-model-id" # Optional: specify your preferred LLM model
            temperature=0.1 # Lower temperature for more factual/less creative answers
        )
        logger.info("QGenie LLM (Chat) model initialized successfully.")

        # Store in session state
        st.session_state['QGENIE_CLIENT'] = qgenie_client
        st.session_state['QGENIE_EMBEDDINGS'] = qgenie_embeddings
        st.session_state['QGENIE_LLM'] = qgenie_llm
        st.session_state['QGENIE_EMAIL_TOOL'] = None

        return True # Indicate success
    except Exception as e:
        logger.error(f"Error initializing QGenie components: {e}")
        st.error(f"Failed to initialize QGenie components. Please check your endpoint, API key, and SDK usage: {e}")
        return False  # Should not be reached if successful or stopped

# --- Data Ingestion using QGenie Tools (Real & Simulated) ---

def _hardcoded_sample_emails(num_emails=8):
    emails_data = {}
    
    # Original email template (slightly enhanced)
    emails_data["sample_email_1.txt"] = """Subject: Project Alpha: UI Design Decision and Next Steps

    Hi Team,

    This is an update regarding the "Alpha Project". We had a meeting on 2023-11-15.
    During the meeting, John Doe proposed a new UI design, and Sarah Lee agreed.
    **Decision:** We will proceed with the new UI design proposed by John.
    **Rationale:** It significantly improves user experience according to initial tests.
    **Key Insight:** Early user feedback indicates high satisfaction with the new flow.
    **Timeline:** Implementation should start next week and conclude by 2024-01-30.
    **People Involved:** John Doe (Lead Designer), Sarah Lee (PM), Engineering Team.

    Please ensure all resources are aligned.

    Best,
    Project Team Lead.
    """
    
    # New email about budget approval
    emails_data["sample_email_2.txt"] = """Subject: URGENT: Budget Approval for Project Alpha

    Dear Management Team,

    I'm writing to request approval for the revised budget for Project Alpha.

    **Decision Needed:** We need to increase the budget by 15% to accommodate the new UI design implementation.
    **Rationale:** The new design requires additional frontend development resources and QA testing.
    **Key Insight:** Our market research shows that this investment will likely increase user engagement by 30%.
    **Timeline:** We need approval by 2023-12-01 to maintain our launch schedule for 2024-01-30.
    **People Involved:** Sarah Lee (PM), Michael Wong (Finance), Executive Team.

    The detailed budget breakdown is attached. Please review and approve at your earliest convenience.

    Thank you,
    Sarah Lee
    Project Manager
    """
    
    # New email about technical challenges
    emails_data["sample_email_3.txt"] = """Subject: Technical Challenges with Database Integration - Project Alpha

    Development Team,

    We've encountered some challenges with the database integration for Project Alpha.

    **Decision:** We will switch from MongoDB to PostgreSQL for the backend.
    **Rationale:** PostgreSQL offers better support for the complex transactions we need to implement.
    **Key Insight:** Initial performance tests show a 40% improvement in query response time.
    **Timeline:** Migration will take approximately 2 weeks, from 2023-12-05 to 2023-12-19.
    **People Involved:** David Chen (Backend Lead), Emily Rodriguez (Database Specialist), QA Team.

    David will organize a technical workshop next Monday to discuss the migration strategy.

    Regards,
    Alex Johnson
    Technical Director
    """
    
    # New email about user testing results
    emails_data["sample_email_4.txt"] = """Subject: User Testing Results for Project Alpha

    Product Team,

    We've completed the first round of user testing for Project Alpha's new UI.

    **Decision:** We will implement the changes suggested by users regarding the checkout flow.
    **Rationale:** 85% of test participants struggled with the current multi-step process.
    **Key Insight:** Simplifying to a single-page checkout could reduce cart abandonment by 25%.
    **Timeline:** Changes to be implemented by 2023-12-15 and retested by 2023-12-22.
    **People Involved:** John Doe (Lead Designer), User Testing Team, Frontend Developers.

    The full testing report is available in Confluence. Key issues are highlighted in red.

    Thanks,
    Lisa Park
    UX Research Lead
    """
    
    # New email about marketing strategy
    emails_data["sample_email_5.txt"] = """Subject: Marketing Strategy for Project Alpha Launch

    Marketing and Product Teams,

    With Project Alpha's launch approaching, we need to finalize our marketing strategy.

    **Decision:** We will focus on digital channels and influencer partnerships for the initial launch.
    **Rationale:** Our target demographic (25-34) is most active on these platforms.
    **Key Insight:** Similar product launches saw 3x better ROI with this approach vs. traditional advertising.
    **Timeline:** Campaign development starts 2024-01-02, with a soft launch on 2024-01-15 and full launch on 2024-01-30.
    **People Involved:** Marketing Team, Sarah Lee (PM), External Agency (CreativeBurst).
    **Projects:** Project Alpha, Q1 Marketing Initiative.

    Please review the attached campaign brief and provide feedback by Friday.

    Best regards,
    James Wilson
    Marketing Director
    """
    
    # New email about compliance review
    emails_data["sample_email_6.txt"] = """Subject: Compliance Review Results - Project Alpha

    Legal and Product Teams,

    The compliance review for Project Alpha has been completed.

    **Decision:** We need to modify the data collection process to comply with GDPR and CCPA.
    **Rationale:** Current implementation doesn't provide clear opt-out mechanisms for users.
    **Key Insight:** Non-compliance could result in fines of up to 4% of annual revenue.
    **Timeline:** Changes must be implemented and verified by 2023-12-20.
    **People Involved:** Legal Team, David Chen (Backend Lead), Privacy Officer.

    I've scheduled a meeting for tomorrow at 2 PM to discuss the specific changes needed.

    Regards,
    Jennifer Lopez
    Legal Counsel
    """
    
    # New email about partnership opportunity
    emails_data["sample_email_7.txt"] = """Subject: Strategic Partnership Opportunity for Project Alpha

    Executive Team,

    TechGiant Inc. has approached us about a potential integration partnership for Project Alpha.

    **Decision Needed:** Whether to pursue this partnership and allocate resources for integration.
    **Rationale:** This could expand our user base by potentially 2 million users.
    **Key Insight:** Similar partnerships have accelerated growth for competitors by 40-50%.
    **Timeline:** TechGiant wants to announce by 2024-02-15 if we proceed.
    **People Involved:** Business Development Team, Sarah Lee (PM), Executive Team.
    **Projects:** Project Alpha, Strategic Partnerships Initiative.

    I've attached the initial proposal from TechGiant and our preliminary analysis.

    Looking forward to discussing this at the next executive meeting.

    Best regards,
    Robert Kim
    Business Development Director
    """
    
    # New email about team expansion
    emails_data["sample_email_8.txt"] = """Subject: Team Expansion for Project Alpha - Hiring Plan

    HR and Management,

    Due to the expanded scope of Project Alpha and the new partnership opportunities, we need to grow the team.

    **Decision:** We will hire 3 additional developers and 1 product analyst in Q1 2024.
    **Rationale:** Current team is at capacity and we need specialized skills for the TechGiant integration.
    **Key Insight:** Internal assessment shows we're understaffed by approximately 30% for the current roadmap.
    **Timeline:** Job postings by 2023-12-10, interviews in January, new hires onboarded by 2024-02-28.
    **People Involved:** HR Team, Sarah Lee (PM), Department Managers.
    **Projects:** Project Alpha, Q1 2024 Hiring Initiative.

    Please review the attached job descriptions and budget allocation.

    Thank you,
    Michelle Thompson
    HR Director
    """
    
    # If num_emails parameter is less than the total available, only return that many
    return {k: emails_data[k] for k in list(emails_data.keys())[:num_emails]}

def fetch_confluence_pages_with_qgenie_tool(num_pages=5):
    """
    Simulates fetching Confluence pages using a QGenie Confluence Tool/API.
    You would integrate a QGenieConfluenceTool or similar here.
    """
    logger.info(f"Simulating fetching {num_pages} Confluence pages...")
    confluence_data = {}
    
    # Original Confluence page template (enhanced)
    confluence_data["confluence_page_1.txt"] = """Title: QGenie Integration Guidelines

    # QGenie Integration Guidelines

    **Project Name:** QGenie Implementation Project
    **Owner:** Alex Chen
    **Last Updated:** 2023-12-01

    This page details the integration of QGenie SDK into our existing systems.
    **Decision:** We will prioritize API-based integration over local model deployment for scalability.
    **Rationale:** Cloud deployment offers better resource elasticity.
    **Key Insight:** Early benchmarks show 20% faster inference with QGenie's cloud endpoints.
    **Timeline:** Phase 1 (Embedding Service) to be completed by 2024-02-15. Phase 2 (LLM Integration) by 2024-03-30.
    **People Involved:** Alex Chen (Tech Lead), Maria Garcia (DevOps), Testing Team.

    Refer to our "Internal API Guidelines" document for more technical details.
    """
    
    # New Confluence page about Project Alpha requirements
    confluence_data["confluence_page_2.txt"] = """Title: Project Alpha - Requirements Specification

    # Project Alpha - Requirements Specification

    **Project Name:** Project Alpha
    **Owner:** Sarah Lee
    **Last Updated:** 2023-11-10

    ## Overview
    Project Alpha aims to revolutionize our customer experience platform with AI-powered personalization.

    ## Key Requirements
    1. Real-time personalization engine
    2. Cross-platform user interface
    3. Integration with existing CRM systems
    4. GDPR and CCPA compliance
    5. Analytics dashboard for business users

    **Decision:** We will use a microservices architecture for better scalability.
    **Rationale:** This allows independent scaling of the personalization engine.
    **Key Insight:** Competitors using monolithic approaches have struggled with peak load times.
    **Timeline:** Requirements finalization by 2023-11-30, development starts 2023-12-05.
    **People Involved:** Sarah Lee (PM), John Doe (Lead Designer), David Chen (Backend Lead), Stakeholders.

    ## Success Criteria
    - 30% improvement in user engagement
    - 25% increase in conversion rate
    - System response time under 200ms
    """
    
    # New Confluence page about technical architecture
    confluence_data["confluence_page_3.txt"] = """Title: Project Alpha - Technical Architecture

    # Project Alpha - Technical Architecture

    **Project Name:** Project Alpha
    **Owner:** David Chen
    **Last Updated:** 2023-11-25

    ## Architecture Overview
    Project Alpha will use a microservices architecture with the following components:

    1. Frontend Layer: React.js with Redux
    2. API Gateway: AWS API Gateway
    3. Personalization Engine: Python/FastAPI microservice
    4. Data Processing: Apache Kafka for event streaming
    5. Database: PostgreSQL for transactional data, MongoDB for user behavior

    **Decision:** We will host on AWS using containerized services with Kubernetes.
    **Rationale:** This provides the best balance of control and managed services.
    **Key Insight:** Initial load testing shows we can handle 10,000 concurrent users.
    **Timeline:** Architecture finalization by 2023-12-10, infrastructure setup by 2023-12-20.
    **People Involved:** David Chen (Backend Lead), Cloud Infrastructure Team, Security Team.

    ## Security Considerations
    - All data in transit and at rest will be encrypted
    - API authentication using OAuth 2.0
    - Regular security audits and penetration testing
    """
    
    # New Confluence page about QGenie API documentation
    confluence_data["confluence_page_4.txt"] = """Title: QGenie API Documentation

    # QGenie API Documentation

    **Project Name:** QGenie Implementation Project
    **Owner:** Maria Garcia
    **Last Updated:** 2023-12-05

    ## API Endpoints

    ### Authentication
    `POST /v1/auth/token`
    Generates an authentication token for API access.

    ### Embedding Service
    `POST /v1/embeddings`
    Generates vector embeddings for provided text.

    ### LLM Service
    `POST /v1/chat/completions`
    Generates text completions or responses based on provided prompts.

    **Decision:** We will implement rate limiting at 100 requests per minute per API key.
    **Rationale:** This balances system load while meeting performance requirements.
    **Key Insight:** Most client applications require no more than 60 requests per minute.
    **Timeline:** API documentation to be completed by 2023-12-15, client libraries by 2024-01-15.
    **People Involved:** Maria Garcia (DevOps), API Documentation Team, Client Library Developers.

    ## Implementation Notes
    - Use the provided SDK rather than direct API calls when possible
    - Cache embeddings for frequently used content
    - Implement exponential backoff for rate limit handling
    """
    
    # New Confluence page about Project Alpha testing strategy
    confluence_data["confluence_page_5.txt"] = """Title: Project Alpha - Testing Strategy

    # Project Alpha - Testing Strategy

    **Project Name:** Project Alpha
    **Owner:** Emily Rodriguez
    **Last Updated:** 2023-12-08

    ## Testing Approach
    Project Alpha will follow a comprehensive testing strategy:

    1. Unit Testing: Jest for frontend, pytest for backend
    2. Integration Testing: Postman collections and automated API tests
    3. Performance Testing: JMeter for load testing
    4. User Acceptance Testing: With selected beta customers
    5. Security Testing: OWASP methodology and automated scans

    **Decision:** We will implement continuous testing in the CI/CD pipeline.
    **Rationale:** This ensures quality at every stage of development.
    **Key Insight:** Early detection of issues reduces fix cost by approximately 5x.
    **Timeline:** Testing framework setup by 2023-12-20, continuous testing by 2024-01-10.
    **People Involved:** Emily Rodriguez (QA Lead), Development Team, Security Team, Beta Users.

    ## Test Environments
    - Development: For developer testing
    - Staging: Mirror of production for final validation
    - Production: Limited rollout with feature flags
    """
    
    # If num_pages parameter is less than the total available, only return that many
    return {k: confluence_data[k] for k in list(confluence_data.keys())[:num_pages]}

def ingest_local_text_files():
    """Ingests local text files. (Now just returns an empty dict)"""
    logger.info("Local text file ingestion disabled as per user request.")
    return {}

# --- Core AI Memory Assistant Functions ---

def summarize_text_with_ai(text, min_length=30, max_length=150):
    """Summarizes text using the QGenie LLM."""
    if not text.strip():
        return ""
    
    # Get LLM from session state
    qgenie_llm = st.session_state.get('QGENIE_LLM')
    if not qgenie_llm:
        logger.error("QGenie LLM not in session state. Cannot summarize.")
        return text[:max_length] + "..." if len(text) > max_length else text

    try:
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert summarizer. Summarize the following text concisely. Focus on key information, decisions, insights, people, projects, and timelines."),
            ("human", "Summarize this text, ensuring the summary is between {min_len} and {max_len} words:\n\n{text}")
        ])
        chain = summary_prompt | qgenie_llm | StrOutputParser()
        # LLMs don't directly take min_length/max_length as a parameter like dedicated summarizers
        # These are now hints in the prompt. Adjust prompt for word count targets.
        summary = chain.invoke({"text": text, "min_len": min_length // 5, "max_len": max_length // 5}) # Convert chars to rough word count
        return summary
    except Exception as e:
        logger.warning(f"Error summarizing text with QGenie LLM: {e}. Returning original text first sentences.")
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
            return ' '.join(sentences[:3])
        except:
            return text[:max_length] + "..." if len(text) > max_length else text

def extract_structured_info_with_llm(text):
    """
    Extracts structured information (decisions, insights, rationales, people, projects, timelines)
    using the QGenie LLM.
    """
    # Get LLM from session state
    qgenie_llm = st.session_state.get('QGENIE_LLM')
    if not qgenie_llm:
        logger.error("QGenie LLM not in session state. Cannot extract structured info.")
        return { "Decision": "N/A", "Rationale": "N/A", "Key Insight": "N/A",
                 "People Involved": "N/A", "Projects": "N/A", "Timeline": "N/A" }

    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert information extractor. Your goal is to extract key details
        from the provided text. If a piece of information is not present, respond with 'N/A' for that field.
        Extract the following exactly in the format specified:
        Decision: [extracted decision or N/A]
        Rationale: [extracted rationale or N/A]
        Key Insight: [extracted key insight or N/A]
        People Involved: [comma-separated names of individuals, or N/A]
        Projects: [comma-separated names of projects/initiatives, or N/A]
        Timeline: [any dates or durations for actions/milestones, or N/A]"""),
        ("human", "Extract information from this text:\n\n{text}")
    ])
    chain = extraction_prompt | qgenie_llm | StrOutputParser()
    extracted_text = chain.invoke({"text": text})

    info = {
        "Decision": "N/A", "Rationale": "N/A", "Key Insight": "N/A",
        "People Involved": "N/A", "Projects": "N/A", "Timeline": "N/A"
    }
    lines = extracted_text.split('\n')
    for line in lines:
        if line.startswith("Decision:"): info["Decision"] = line[len("Decision:"):].strip()
        elif line.startswith("Rationale:"): info["Rationale"] = line[len("Rationale:"):].strip()
        elif line.startswith("Key Insight:"): info["Key Insight"] = line[len("Key Insight:"):].strip()
        elif line.startswith("People Involved:"): info["People Involved"] = line[len("People Involved:"):].strip()
        elif line.startswith("Projects:"): info["Projects"] = line[len("Projects:"):].strip()
        elif line.startswith("Timeline:"): info["Timeline"] = line[len("Timeline:"):].strip()
    return info

def build_memory_graph(documents_dict):
    """
    Builds a NetworkX graph using QGenie for NER and LLM for structured extraction.
    """
    G = nx.Graph()
    
    # Get components from session state
    qgenie_client = st.session_state.get('QGENIE_CLIENT')
    qgenie_llm = st.session_state.get('QGENIE_LLM')
    
    if not qgenie_client or not qgenie_llm:
        st.error("QGenie components not in session state. Cannot build graph.")
        return G

    for doc_name, full_text in documents_dict.items():
        doc_type = 'email' if doc_name.startswith('sample_email') else ('confluence' if doc_name.startswith('confluence_page') else 'document')
        G.add_node(doc_name, type=doc_type, content=full_text, label=doc_name)

        # 1. General Named Entity Recognition (using QGenieClient's direct method, if available, else LLM)
        try:
            # Assuming QGenieClient has a direct 'extract_named_entities' method
            ner_results = qgenie_client.extract_named_entities(text=full_text)
            extracted_entities_raw = ner_results if isinstance(ner_results, list) else []
            logger.debug(f"NER results from QGenieClient for {doc_name}: {extracted_entities_raw}")
        except AttributeError:
            logger.warning("QGenieClient has no 'extract_named_entities'. Falling back to LLM prompt for NER (may be less structured).")
            ner_prompt = ChatPromptTemplate.from_messages([
                ("system", "Identify and list all distinct named entities (e.g., persons, organizations, locations, project names, dates) in the following text. Respond with a comma-separated list of entities."),
                ("human", "Text:\n\n{text}")
            ])
            chain = ner_prompt | qgenie_llm | StrOutputParser()  # Use qgenie_llm from session state
            ner_output = chain.invoke({"text": full_text})
            # This parsing is simplistic; real NER output is structured
            extracted_entities_raw = [{'entity_type': 'MISC', 'text': e.strip()} for e in re.split(r'[,\n;]', ner_output) if e.strip()]
            logger.debug(f"NER results from LLM for {doc_name}: {extracted_entities_raw}")
        except Exception as e:
            logger.warning(f"Error performing NER on {doc_name}: {e}")
            extracted_entities_raw = []

        for entity_data in extracted_entities_raw:
            entity_name = entity_data.get('text', '').strip()
            entity_type = entity_data.get('entity_type', 'MISC').upper()
            if len(entity_name) > 1 and not re.match(r'^[^\w\s]+$', entity_name):
                # Ensure unique ID for entity nodes by type and name
                node_id = f"ENTITY_{entity_type}_{entity_name}"
                if not G.has_node(node_id):
                    G.add_node(node_id, type=f'entity_{entity_type}', entity_category=entity_type, label=entity_name)
                G.add_edge(doc_name, node_id, relation='mentions')

        # 2. Structured Information Extraction with LLM (Decisions, Insights, etc.)
        try:
            structured_info = extract_structured_info_with_llm(full_text)  # This function should also use qgenie_llm from session state
            for key, value in structured_info.items():
                if value and value != "N/A":
                    # Create a unique node ID for structured info if it's substantial
                    if len(value) > 50: # Truncate for ID, keep full for content
                        node_value_id = value[:47] + "..."
                    else:
                        node_value_id = value
                    
                    node_id = f"EXTRACT_{key.replace(' ', '_').upper()}_{node_value_id}"
                    
                    if not G.has_node(node_id):
                        G.add_node(node_id, type=key.replace(" ", "_"), content=value, label=node_value_id)
                    G.add_edge(doc_name, node_id, relation=f"has_{key.lower().replace(' ', '_')}")

                    # Link People, Projects, Timelines to their respective Decision/Insight/Document
                    if key in ["People Involved", "Projects", "Timeline"]:
                        items = [item.strip() for item in value.split(',') if item.strip()]
                        for item in items:
                            item_node_id = f"ENTITY_{key.replace(' ', '_').upper()}_{item}"
                            if not G.has_node(item_node_id):
                                # Better guess entity type for these fields
                                if key == "People Involved": item_type = 'entity_PERSON'
                                elif key == "Projects": item_type = 'entity_PROJECT'
                                elif key == "Timeline": item_type = 'entity_DATE'
                                else: item_type = 'entity_MISC'
                                G.add_node(item_node_id, type=item_type, entity_category=item_type.split('_')[-1], label=item)
                            G.add_edge(node_id, item_node_id, relation=f"is_part_of_{key.lower().replace(' ', '_')}")
                            # Also link people/projects directly to the document for broader context
                            if key in ["People Involved", "Projects"]:
                                G.add_edge(doc_name, item_node_id, relation=f"mentions_{key.lower().replace(' ', '_')}")

        except Exception as e:
            logger.warning(f"Error extracting structured info for {doc_name} with LLM: {e}")
    
    return G

def retrieve_and_generate_answer(query, documents_dict, k=3):
    """
    Retrieves relevant document chunks and generates an answer using the QGenie LLM.
    """
    # Add debug logging to see what's happening
    logger.info(f"retrieve_and_generate_answer called with query: {query[:30]}...")
    
    # Get components from session state
    qgenie_embeddings = st.session_state.get('QGENIE_EMBEDDINGS')
    qgenie_llm = st.session_state.get('QGENIE_LLM')
    
    logger.info(f"qgenie_embeddings from session state is None: {qgenie_embeddings is None}")
    logger.info(f"qgenie_llm from session state is None: {qgenie_llm is None}")
    
    if not qgenie_embeddings or not qgenie_llm:
        logger.error("QGenie components not in session state. Cannot answer questions.")
        return "QGenie components not initialized. Cannot answer questions.", []

    if not documents_dict or not query.strip():
        return "Please load data first or enter a query.", []

    # Prepare retriever: Combine all document texts and names for embedding
    texts = list(documents_dict.values())
    metadatas = [{"source": name} for name in documents_dict.keys()]

    try:
        # If QGenieLLM doesn't have embed_documents method, use a workaround
        if hasattr(qgenie_embeddings, 'embed_documents'):
            doc_embeddings = qgenie_embeddings.embed_documents(texts)
        else:
            # Fallback: Use the LLM to generate embeddings one by one
            logger.warning("Using fallback method for document embeddings")
            doc_embeddings = []
            for text in texts:
                # Use a simple prompt to get embeddings via the LLM
                embedding_prompt = f"Generate a semantic embedding for this text: {text[:1000]}..."
                # This is a simplified approach - in a real implementation, you'd need a proper embedding method
                response = qgenie_embeddings.invoke(embedding_prompt)
                # For demonstration purposes, we're creating random embeddings
                # In a real implementation, you'd extract actual embeddings from the LLM response
                import numpy as np
                doc_embeddings.append(np.random.rand(768))  # Typical embedding dimension
    except Exception as e:
        logger.error(f"Error embedding documents for RAG: {e}")
        return "Error in embedding documents for retrieval.", []

    try:
        # If QGenieLLM doesn't have embed_query method, use a workaround
        if hasattr(qgenie_embeddings, 'embed_query'):
            query_embedding = qgenie_embeddings.embed_query(query)
        else:
            # Fallback: Use the LLM to generate an embedding for the query
            logger.warning("Using fallback method for query embedding")
            embedding_prompt = f"Generate a semantic embedding for this query: {query}"
            # This is a simplified approach - in a real implementation, you'd need a proper embedding method
            response = qgenie_embeddings.invoke(embedding_prompt)
            # For demonstration purposes, we're creating a random embedding
            # In a real implementation, you'd extract an actual embedding from the LLM response
            import numpy as np
            query_embedding = np.random.rand(768)  # Typical embedding dimension
    except Exception as e:
        logger.error(f"Error embedding query for RAG: {e}")
        return "Error in embedding query for retrieval.", []

    # Calculate cosine similarity
    doc_embeddings_tensor = torch.tensor(doc_embeddings).to(torch.float32)
    query_embedding_tensor = torch.tensor(query_embedding).unsqueeze(0).to(torch.float32)

    cosine_scores = torch.cosine_similarity(query_embedding_tensor, doc_embeddings_tensor)

    # Get top_k results
    top_results_indices = torch.argsort(cosine_scores, descending=True)[:k]
    
    retrieved_docs_with_scores = []
    for idx in top_results_indices:
        sim_score = cosine_scores[idx].item()
        if sim_score > 0.3: # Minimum similarity threshold
            retrieved_docs_with_scores.append({
                "document": metadatas[idx]["source"],
                "content": texts[idx],
                "similarity": sim_score
            })
    
    if not retrieved_docs_with_scores:
        return "No relevant documents found for your query in the memory graph.", []

    # Prepare context for the LLM
    context_string = "\n\n".join([f"Document: {doc['document']}\nContent: {doc['content']}" for doc in retrieved_docs_with_scores])
    
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant, Q-MATE, specialized in providing precise answers based only on the provided context about organizational knowledge (emails, confluence pages, documents).
        If the answer is not in the context, clearly state "I don't have enough information from the provided context to answer that."
        Always cite the specific document(s) (e.g., 'email_abc1.txt', 'confluence_page_1.txt') from which you extracted the answer.
        Focus on decisions, key insights, rationales, people involved, projects, and timelines relevant to the question."""),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    rag_chain = rag_prompt | qgenie_llm | StrOutputParser()

    try:
        answer = rag_chain.invoke({"context": context_string, "question": query})
    except Exception as e:
        logger.error(f"Error generating answer with QGenie LLM: {e}")
        return "Error generating answer from retrieved context.", []

    return answer, retrieved_docs_with_scores

# ... existing code ...

def create_focused_graph(memory_graph, query, answer, retrieved_contexts):
    """
    Creates a highly focused graph based on the query and answer,
    showing only the most relevant entities and relationships.
    
    Args:
        memory_graph: The full memory graph
        query: The user's query
        answer: The AI's answer
        retrieved_contexts: The documents retrieved for answering
        
    Returns:
        focused_graph: A clean NetworkX graph with only the most relevant entities
        project_info: Dictionary with project details
        context_panels: Dictionary with contextual information for UI panels
    """
    # Initialize return values
    focused_graph = nx.Graph()
    project_info = {
        "name": "Project",
        "status": "Active",
        "key_decisions": [],
        "contributors": []
    }
    context_panels = {
        "recent_decisions": [],
        "contributors": [],
        "key_insights": []
    }
    
    # Get LLM from session state for entity extraction
    qgenie_llm = st.session_state.get('QGENIE_LLM')
    if not qgenie_llm:
        logger.error("QGenie LLM not in session state. Cannot create focused graph.")
        return focused_graph, project_info, context_panels
    
    try:
        # 1. Extract ONLY the most important entities from query and answer
        entity_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract ONLY the 4-6 most important entities from the query and answer.
            Focus on the main people, projects, and technical concepts that are DIRECTLY mentioned.
            Format as a simple comma-separated list with no additional text.
            DO NOT include generic terms or concepts that aren't specifically named in the text."""),
            ("human", "Query: {query}\nAnswer: {answer}")
        ])
        
        chain = entity_prompt | qgenie_llm | StrOutputParser()
        entities_text = chain.invoke({"query": query, "answer": answer})
        key_entities = [e.strip() for e in entities_text.split(',') if e.strip()]
        
        # 2. Extract project information
        project_prompt = ChatPromptTemplate.from_messages([
            ("system", """Based on the query and answer, identify:
            1. The exact project name
            2. The current project status
            3. 2-3 key decisions related to this project (be specific)
            
            Format as:
            Project: [project name]
            Status: [status]
            Decision 1: [first decision]
            Decision 2: [second decision]
            Decision 3: [third decision]"""),
            ("human", "Query: {query}\nAnswer: {answer}")
        ])
        
        chain = project_prompt | qgenie_llm | StrOutputParser()
        project_text = chain.invoke({"query": query, "answer": answer})
        
        # Parse project information
        for line in project_text.split('\n'):
            if line.startswith("Project:"):
                project_info["name"] = line[len("Project:"):].strip()
            elif line.startswith("Status:"):
                project_info["status"] = line[len("Status:"):].strip()
            elif line.startswith("Decision"):
                decision = line.split(":", 1)[1].strip() if ":" in line else line
                if decision and decision != "Unknown":
                    project_info["key_decisions"].append(decision)
        
        # 3. Extract contributors and decisions for panels
        panel_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract the following from the answer:
            1. List of contributors/people mentioned (with their roles if available)
            2. Recent decisions mentioned
            3. Key technical components mentioned
            
            Format as:
            Contributors: [person1 (role)], [person2 (role)], ...
            Decisions: [decision1], [decision2], ...
            Components: [component1], [component2], ..."""),
            ("human", "Answer: {answer}")
        ])
        
        chain = panel_prompt | qgenie_llm | StrOutputParser()
        panel_text = chain.invoke({"answer": answer})
        
        # Parse panel information
        for line in panel_text.split('\n'):
            if line.startswith("Contributors:"):
                contributors_text = line[len("Contributors:"):].strip()
                context_panels["contributors"] = [c.strip() for c in contributors_text.split(',') if c.strip()]
            elif line.startswith("Decisions:"):
                decisions_text = line[len("Decisions:"):].strip()
                context_panels["recent_decisions"] = [d.strip() for d in decisions_text.split(',') if d.strip()]
            elif line.startswith("Components:"):
                components_text = line[len("Components:"):].strip()
                context_panels["key_insights"] = [c.strip() for c in components_text.split(',') if c.strip()]
        
        # 4. Build the focused graph directly from extracted entities
        # Add entity nodes
        for entity in key_entities:
            # Determine node type based on entity name
            if any(name.lower() in entity.lower() for name in ["raj", "sayali", "abhishek", "avinash", "prashanth", "rajashree"]):
                node_type = "entity_PERSON"
            elif "project" in entity.lower() or "molokai" in entity.lower():
                node_type = "entity_PROJECT"
            elif any(term.lower() in entity.lower() for term in ["sensor", "testing", "ois", "pd", "als", "prox", "persist", "bin"]):
                node_type = "entity_COMPONENT"
            else:
                node_type = "entity_MISC"
                
            focused_graph.add_node(entity, type=node_type, label=entity)
        
        # 5. Generate relationships between entities using LLM
        # This is the key part to ensure we have edges in the graph
        if len(key_entities) >= 2:  # Need at least 2 entities to form relationships
            relationship_prompt = ChatPromptTemplate.from_messages([
                ("system", """For each pair of entities below, determine if there is a direct relationship based on the answer.
                Format each relationship as:
                Entity1|relationship|Entity2
                
                For example:
                Raj|worked on|Project Molokai
                Sayali Ayarkar|supports|Persist Bin Generation
                
                Only include relationships that are clearly implied in the answer.
                Be specific about the relationship type.
                Provide at least 3-5 relationships if possible."""),
                ("human", "Entities: {entities}\nAnswer: {answer}")
            ])
            
            chain = relationship_prompt | qgenie_llm | StrOutputParser()
            relationships_text = chain.invoke({"entities": ", ".join(key_entities), "answer": answer})
            
            # Parse and add relationships as edges
            for line in relationships_text.split('\n'):
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        source = parts[0].strip()
                        relation = parts[1].strip()
                        target = parts[2].strip()
                        
                        # Check if both entities exist in our nodes
                        source_match = next((node for node in focused_graph.nodes() if source.lower() in node.lower()), None)
                        target_match = next((node for node in focused_graph.nodes() if target.lower() in node.lower()), None)
                        
                        if source_match and target_match and source_match != target_match:
                            # Add the edge with the relationship as a label
                            focused_graph.add_edge(source_match, target_match, relation=relation)
        
        # 6. If we still don't have enough edges, create some basic connections
        if focused_graph.number_of_edges() < 2 and len(key_entities) >= 3:
            # Find project node
            project_node = next((node for node in focused_graph.nodes() if "project" in node.lower() or "molokai" in node.lower()), None)
            
            if project_node:
                # Connect people to the project
                for node in focused_graph.nodes():
                    if focused_graph.nodes[node].get('type') == "entity_PERSON" and not focused_graph.has_edge(node, project_node):
                        focused_graph.add_edge(node, project_node, relation="involved in")
                
                # Connect components to the project
                for node in focused_graph.nodes():
                    if focused_graph.nodes[node].get('type') == "entity_COMPONENT" and not focused_graph.has_edge(node, project_node):
                        focused_graph.add_edge(node, project_node, relation="part of")
        
        # 7. Add contributors to project_info
        project_info["contributors"] = context_panels["contributors"]
        
    except Exception as e:
        logger.error(f"Error creating focused graph: {e}")
    
    return focused_graph, project_info, context_panels

def render_simplified_graph(G, query, answer):
    """
    Renders a clean, simplified graph that's easy to understand.
    """
    st.subheader("Relationship Graph")
    
    # Create a PyVis network
    net = Network(height="400px", width="100%", bgcolor="#ffffff", font_color="#000000")
    
    # Configure physics for better layout
    net.barnes_hut(gravity=-2000, central_gravity=0.3, spring_length=200)
    
    # Set options for a cleaner look
    net.set_options("""
    var options = {
      "nodes": {
        "font": {
          "size": 18,
          "face": "Arial"
        },
        "borderWidth": 2,
        "shadow": true
      },
      "edges": {
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 1
          }
        },
        "color": "#000000",
        "font": {
          "size": 14,
          "align": "middle"
        },
        "smooth": true
      },
      "physics": {
        "stabilization": {
          "iterations": 100
        }
      }
    }
    """)
    
    # Add nodes with distinct styling
    for node, data in G.nodes(data=True):
        node_type = data.get('type', 'unknown')
        
        if node_type == 'entity':
            # Check if this is a person
            if any(name.lower() in node.lower() for name in ["raj", "sayali", "john", "sarah", "david", "emily", "michael"]):
                color = "#4169E1"  # Royal Blue for people
                shape = "circularImage"
                image = "https://cdn.pixabay.com/photo/2016/08/08/09/17/avatar-1577909_960_720.png"
            # Check if this is a project
            elif any(term.lower() in node.lower() for term in ["project", "molokai", "alpha", "initiative"]):
                color = "#20B2AA"  # Light Sea Green for projects
                shape = "box"
                image = ""
            # Check if this is a technology/component
            elif any(term.lower() in node.lower() for term in ["sensor", "testing", "detection", "api", "database"]):
                color = "#9932CC"  # Dark Orchid for technologies
                shape = "diamond"
                image = ""
            else:
                color = "#FFA500"  # Orange for other entities
                shape = "dot"
                image = ""
                
            net.add_node(
                node,
                label=node,
                color=color,
                shape=shape,
                image=image,
                size=30,
                borderWidth=2,
                shadow=True
            )
        elif node_type == 'document':
            net.add_node(
                node,
                label=node,
                color="#A9A9A9",  # Dark Gray for documents
                shape="file",
                size=25,
                borderWidth=1
            )
    
    # Add edges with labels
    for u, v, data in G.edges(data=True):
        relation = data.get('relation', '')
        
        # Only show relation labels between entities, not for document connections
        if G.nodes[u].get('type') == 'entity' and G.nodes[v].get('type') == 'entity':
            net.add_edge(u, v, title=relation, label=relation, font={'align': 'middle'})
        else:
            net.add_edge(u, v, title=relation)
    
    # Save and display the graph
    try:
        net.save_graph("focused_graph.html")
        with open("focused_graph.html", 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=450, scrolling=False)
    except Exception as e:
        st.error(f"Failed to display graph: {e}")

def render_enhanced_graph_ui(focused_graph, project_info, context_panels, query, answer):
    """
    Renders an enhanced UI with a clean graph visualization and contextual panels,
    similar to the example mockup.
    """
    # Create columns for the layout
    left_col, right_col = st.columns([1, 2])
    
    with left_col:
        # Recent decisions panel
        st.subheader("Recent decisions")
        if context_panels.get("recent_decisions"):
            for decision in context_panels["recent_decisions"][:3]:  # Show top 3
                st.markdown(f"✅ {decision[:50]}..." if len(decision) > 50 else f"✅ {decision}")
        else:
            st.info("No recent decisions found")
        
        # Contributors panel
        st.subheader("Contributors")
        if context_panels.get("contributors"):
            for contributor in context_panels["contributors"]:
                # Check if contributor has left/departed based on answer text
                if "departed" in answer.lower() and contributor.lower() in answer.lower() and "left" in answer.lower():
                    st.markdown(f"{contributor} (Departed)")
                else:
                    st.markdown(contributor)
        else:
            st.info("No contributors found")
    
    with right_col:
        # Project overview panel
        st.subheader(f"{project_info.get('name', 'Project')} Overview")
        
        st.markdown(f"**Status:** {project_info.get('status', 'Unknown')}")
        
        st.markdown("**Key Decisions:**")
        if project_info.get("key_decisions"):
            for decision in project_info.get("key_decisions")[:3]:
                st.markdown(f"• {decision}")
        else:
            st.markdown("No key decisions found")
        
        st.markdown("**Contributors:**")
        if project_info.get("contributors"):
            contributors_text = ", ".join(project_info.get("contributors")[:5])
            st.markdown(f"• {contributors_text}")
        
        # Generate and display the graph
        if focused_graph and focused_graph.number_of_nodes() > 0:
            st.markdown("### Related entities")
            
            # Convert the graph to PyVis format
            net = Network(height="300px", width="100%", bgcolor="#f9f9f9", font_color="#333333", cdn_resources='remote', directed=False)
            
            # Use a more organized layout
            net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=150, spring_strength=0.05)
            
            # Set options for a cleaner look
            net.set_options("""
            var options = {
              "nodes": {
                "font": {
                  "size": 16,
                  "face": "Arial"
                },
                "borderWidth": 2,
                "shadow": true
              },
              "edges": {
                "color": {
                  "inherit": false
                },
                "width": 1.5,
                "smooth": {
                  "type": "continuous"
                }
              },
              "physics": {
                "stabilization": {
                  "iterations": 100
                },
                "minVelocity": 0.75
              }
            }
            """)
            
            # Add nodes with clean styling
            for node, data in focused_graph.nodes(data=True):
                node_label = data.get('label', node)
                node_type = data.get('type', '')
                
                # Determine node color and size based on type
                if node_type.startswith('entity_PERSON'):
                    color = "#4169E1"  # Royal Blue for people
                    size = 25
                elif "project" in node_label.lower() or node_type.startswith('entity_PROJECT'):
                    color = "#20B2AA"  # Light Sea Green for projects
                    size = 25
                elif node_type == 'Decision':
                    color = "#DC143C"  # Crimson for decisions
                    size = 22
                elif node_type.startswith('entity_'):
                    color = "#9932CC"  # Dark Orchid for other entities
                    size = 20
                else:
                    color = "#778899"  # Light Slate Gray for misc
                    size = 18
                
                # Truncate long labels
                if len(node_label) > 20:
                    display_label = node_label[:17] + "..."
                else:
                    display_label = node_label
                
                # Create tooltip
                tooltip = f"{node_label}"
                if 'content' in data and data['content']:
                    tooltip += f"\n{data['content'][:100]}..."
                
                # Add node
                net.add_node(
                    node, 
                    label=display_label,
                    title=tooltip,
                    color=color,
                    size=size,
                    borderWidth=2,
                    shadow=True
                )
            
            # Add edges
            for u, v, data in focused_graph.edges(data=True):
                relation = data.get('relation', 'related').replace('_', ' ')
                if len(relation) > 15:
                    relation = relation[:12] + "..."
                
                net.add_edge(
                    u, v,
                    title=relation,
                    color="#555555",
                    width=1.5
                )
            
            # Save and display the graph
            try:
                net.save_graph(GRAPH_HTML_PATH)
                with open(GRAPH_HTML_PATH, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=350, scrolling=False)
            except Exception as e:
                st.error(f"Failed to display graph: {e}")
        else:
            st.info("No entity relationships found to visualize")

@st.cache_data(show_spinner="Loading and processing data...")
def load_and_process_all_data():
    """
    Loads all data sources, processes with QGenie components, and builds the memory graph.
    """
    # 1. Initialize QGenie components (only if not already global)
    if 'QGENIE_CLIENT' not in st.session_state:
        if not load_qgenie_components():
            return {}, {}, nx.Graph(), []  # If initialization failed, stop

    # 2. Ingest data from various sources
    ingested_local_docs = ingest_local_text_files()

    # 3. Use hardcoded sample emails instead of fetching via Gmail API
    st.info("Ingesting hardcoded sample emails...")
    ingested_emails = _hardcoded_sample_emails(num_emails=8)
    st.info(f"Ingested {len(ingested_emails)} sample emails.")

    # 4. Fetch Confluence pages (simulated)
    st.info("Fetching Confluence pages via QGenie tool (simulated)...")
    ingested_confluence = fetch_confluence_pages_with_qgenie_tool(num_pages=5)
    st.info(f"Ingested {len(ingested_confluence)} Confluence pages.")

    # Combine all data
    all_ingested_data = {**ingested_local_docs, **ingested_emails, **ingested_confluence}

    if not all_ingested_data:
        st.warning("No data found from any source. Please check paths/tool configurations.")
        return {}, {}, nx.Graph(), []

    # 5. Summarize all documents
    st.info("Generating summaries using QGenie LLM...")
    summaries = {
        filename: summarize_text_with_ai(content)
        for filename, content in all_ingested_data.items()
    }

    # 6. Build memory graph with rich extraction
    st.info("Building rich memory graph with NER and LLM-based structured extraction...")
    memory_graph = build_memory_graph(all_ingested_data)

    # 7. Extract all unique entities for the sidebar
    all_entities = sorted(
        list(
            set(
                n
                for n, data in memory_graph.nodes(data=True)
                if data.get("type", "").startswith("entity_")
                or data.get("type") in ["Decision", "Key_Insight", "Rationale", "Project"]
            )
        )
    )

    st.success(f"Successfully processed {len(all_ingested_data)} total documents with AI.")
    return all_ingested_data, summaries, memory_graph, all_entities

# --- Streamlit Application ---

st.set_page_config(layout="wide", page_title="Q-MATE: Don't Worry Buddy I Got It All")

st.title("Q-MATE:Don't Worry Buddy I Got It All")

# --- Initialize session state variables ---
if 'data_loaded' not in st.session_state: st.session_state['data_loaded'] = False
if 'ingested_data' not in st.session_state: st.session_state['ingested_data'] = {}
if 'summaries' not in st.session_state: st.session_state['summaries'] = {}
if 'memory_graph' not in st.session_state: st.session_state['memory_graph'] = nx.Graph()
if 'all_entities' not in st.session_state: st.session_state['all_entities'] = []
if 'last_llm_answer' not in st.session_state: st.session_state['last_llm_answer'] = ""
if 'search_results' not in st.session_state: st.session_state['search_results'] = []
if 'recent_searches' not in st.session_state: st.session_state['recent_searches'] = []
if 'focused_graph' not in st.session_state: st.session_state['focused_graph'] = nx.Graph()
if 'project_info' not in st.session_state: st.session_state['project_info'] = {}
if 'context_panels' not in st.session_state: st.session_state['context_panels'] = {}

# --- Sidebar for Data Ingestion and Configuration ---
st.sidebar.header("Configuration")

# Use a button to explicitly load data
if st.sidebar.button("Load/Reload All Data (AI Agent)", key="load_button") or not st.session_state['data_loaded']:
    st.session_state['ingested_data'], st.session_state['summaries'], \
    st.session_state['memory_graph'], st.session_state['all_entities'] = \
        load_and_process_all_data()
    if st.session_state['ingested_data']:
        st.session_state['data_loaded'] = True
    else:
        st.session_state['data_loaded'] = False

# --- Main Content Area ---

tab1, tab2 = st.tabs(["AI Agent & Memory Graph", "Document Summaries"])

with tab1:
    st.header("AI Agent: Ask Questions & Explore Knowledge Graph")
    query = st.text_input("Ask Q-MATE a question:", key="agent_query_input")
    search_button_col, _ = st.columns([0.15, 0.85])
    with search_button_col:
        ask_triggered = st.button("Ask Q-MATE", key="ask_button")

    if ask_triggered:
        if query and st.session_state['ingested_data']:
            with st.spinner("Q-MATE is thinking..."):
                # Reset the focused graph
                st.session_state['focused_graph'] = nx.Graph()
                st.session_state['project_info'] = {}
                st.session_state['context_panels'] = {}
                
                answer, retrieved_contexts = retrieve_and_generate_answer(query, st.session_state['ingested_data'])
                st.session_state['last_llm_answer'] = answer
                st.session_state['search_results'] = retrieved_contexts
                
                # Create a focused graph and extract contextual information
                if st.session_state['memory_graph'].number_of_nodes() > 0:
                    focused_graph, project_info, context_panels = create_focused_graph(
                        st.session_state['memory_graph'], 
                        query, 
                        answer, 
                        retrieved_contexts
                    )
                    st.session_state['focused_graph'] = focused_graph
                    st.session_state['project_info'] = project_info
                    st.session_state['context_panels'] = context_panels
                
                st.session_state['recent_searches'].insert(0, {
                    'query': query,
                    'time': datetime.datetime.now().strftime("%H:%M:%S"),
                    'answer_start': answer[:100] + "..." if len(answer) > 100 else answer
                })
                st.session_state['recent_searches'] = st.session_state['recent_searches'][:5]
        else:
            st.warning("Please load data first or enter a question.")
            st.session_state['last_llm_answer'] = ""
            st.session_state['search_results'] = []
            st.session_state['focused_graph'] = nx.Graph()
            st.session_state['project_info'] = {}
            st.session_state['context_panels'] = {}

    # Display the answer and enhanced UI
    if st.session_state.get('data_loaded') and st.session_state['ingested_data']:
        if st.session_state['last_llm_answer']:
            # Create a container for the entire response
            with st.container():
                # Answer section
                st.subheader("Q-MATE's Answer")
                st.write(st.session_state['last_llm_answer'])
                
                # Enhanced graph UI with contextual panels
                st.markdown("---")
                
                # Use the enhanced UI renderer
                render_enhanced_graph_ui(
                    st.session_state.get('focused_graph', nx.Graph()),
                    st.session_state.get('project_info', {}),
                    st.session_state.get('context_panels', {}),
                    query if 'query' in locals() else "",
                    st.session_state['last_llm_answer']
                )
                
                # Show sources used
                st.markdown("---")
                st.subheader("Sources Used")
                if st.session_state['search_results']:
                    for i, result in enumerate(st.session_state['search_results'][:3]):  # Limit to top 3
                        st.markdown(f"**{i+1}. {result['document']}** (Relevance: {result['similarity']:.2f})")
                        st.markdown(f"*{summarize_text_with_ai(result['content'], max_length=100)}*")
                else:
                    st.info("No specific sources were used to generate this answer.")
        elif ask_triggered and not st.session_state['last_llm_answer']:
            st.info("Q-MATE could not generate an answer based on the available data.")
        else:
            st.info("Ask Q-MATE a question to get an answer here.")
    else:
        st.info("Load data from the sidebar to enable the AI Agent.")

with tab2:
    st.header("Document Summaries (QGenie LLM Summarization)")
    if st.session_state.get('data_loaded') and st.session_state['summaries']:
        selected_doc = st.selectbox("Select a document to view its summary:",
                                    list(st.session_state['summaries'].keys()), key="summary_select_box")
        if selected_doc:
            original_text = st.session_state['ingested_data'].get(selected_doc, "Original text not available.")
            summary_text = st.session_state['summaries'].get(selected_doc, "*No summary generated for this document.*")

            st.subheader(f"Summary for {selected_doc}")
            st.write(summary_text)

            with st.expander("View Original Document"):
                st.text_area("Original Content", original_text, height=300, key="original_text_area")
    else:
        st.info("No summaries available. Please load and process data from the sidebar first.")

# --- Enhanced Sidebar Content ---
st.sidebar.markdown("---")
st.sidebar.subheader("Recent Questions")
if st.session_state['recent_searches']:
    for i, search_item in enumerate(st.session_state['recent_searches']):
        st.sidebar.markdown(f"**{i+1}. {search_item['time']}**")
        st.sidebar.markdown(f"Question: `{search_item['query']}`")
        st.sidebar.markdown(f"Answer: `{search_item['answer_start']}`")
        if i < len(st.session_state['recent_searches']) - 1:
            st.sidebar.markdown("---")
else:
    st.sidebar.info("No recent questions yet.")

st.sidebar.markdown("---")
st.sidebar.subheader("Key Entities & Extractions")
if st.session_state.get('data_loaded') and st.session_state['all_entities']:
    entities_to_show = st.session_state['all_entities'][:50]
    for entity in entities_to_show:
        st.sidebar.markdown(f"- **{entity}**")
    if len(st.session_state['all_entities']) > 50:
        st.sidebar.markdown(f"*(and {len(st.session_state['all_entities']) - 50} more...)*")
else:
    st.sidebar.info("Load data to see entities and extractions.")