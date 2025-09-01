import os
import re
import nltk
import streamlit as st
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pyvis.network import Network
import datetime # For timestamping recent searches

# --- Configuration ---
# IMPORTANT: Update this to your actual directory!
DEFAULT_DATA_DIRECTORY = r"C:\shanmitha\sample_files" # Ensure this path exists and has .txt files for testing
GRAPH_HTML_PATH = "graph.html" # Path to save the PyVis graph HTML

# --- NLTK Downloads ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# --- Core AI Memory Assistant Functions (No major changes here from previous fix) ---

def ingest_text_files(directory_path):
    """
    Ingests text files from a specified directory.
    """
    data = {}
    if not os.path.isdir(directory_path):
        st.error(f"Error: Directory not found at {directory_path}")
        return data

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data[filename] = f.read()
            except Exception as e:
                st.warning(f"Error reading {filename}: {e}")
    return data

def summarize_text(text, num_sentences=3):
    """
    Summarizes a given text by extracting the most important sentences using TF-IDF.
    """
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    except Exception:
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.!?])\s+', text)

    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) == 0:
        return ""
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)

    vectorizer = TfidfVectorizer()
    scores = []
    top_sentence_indices = []

    try:
        X = vectorizer.fit_transform(sentences)
        scores = X.sum(axis=1).A1
        top_sentence_indices = scores.argsort()[-num_sentences:]
    except ValueError as e:
        st.warning(f"Warning: Could not perform TF-IDF for summary. Likely empty sentences list or no meaningful tokens. Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during TF-IDF vectorization for summary: {e}")

    # FIX: Use len() to check if the numpy array has elements
    if len(top_sentence_indices) > 0:
        valid_indices = sorted([idx for idx in top_sentence_indices if idx < len(sentences)])
        ranked_sentences = [sentences[i] for i in valid_indices]
        return ' '.join(ranked_sentences)
    else:
        st.info("No meaningful sentences found for TF-IDF summary. Returning first available sentences or empty string.")
        return ' '.join(sentences[:num_sentences])

def build_memory_graph(summaries):
    """
    Builds a NetworkX graph representing documents and extracted entities.
    """
    G = nx.Graph()
    for doc, summary in summaries.items():
        G.add_node(doc, type='document')
        potential_entities = re.findall(r'\b[A-Z][a-z]*(?:\s[A-Z][a-z]*)*\b', summary)

        for entity in set(potential_entities):
            if len(entity.split()) > 1 or (len(entity) > 1 and entity.lower() not in ["the", "and", "but", "for", "nor", "or", "so", "yet", "a", "an", "is", "of", "in", "to"]):
                G.add_node(entity, type='entity')
                G.add_edge(doc, entity, relation='mentions')
    return G

def search_documents(query, documents, top_k=3):
    """
    Searches for documents most similar to a given query using TF-IDF and cosine similarity.
    """
    if not documents or not query.strip():
        return []

    doc_names = list(documents.keys())
    doc_texts = list(documents.values())

    vectorizer = TfidfVectorizer()
    try:
        X = vectorizer.fit_transform(doc_texts + [query])
        similarities = cosine_similarity(X[-1], X[:-1]).flatten()
    except ValueError:
        st.warning("Warning: Could not vectorize documents or query for search. Returning empty results.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred during search vectorization: {e}")
        return []

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        if 0 <= idx < len(doc_names) and similarities[idx] > 0:
            results.append({
                'document': doc_names[idx],
                'similarity': similarities[idx],
                'excerpt': summarize_text(doc_texts[idx], num_sentences=2)
            })
    return results

def prepare_ui_data(graph, search_results):
    """
    Prepares graph and search result data for UI display.
    """
    nodes = [{'id': n, 'type': graph.nodes[n]['type']} for n in graph.nodes]
    edges = [{'source': u, 'target': v, 'relation': graph.edges[u, v]['relation']} for u, v in graph.edges]

    sidebar = [{'document': r['document'], 'excerpt': r['excerpt']} for r in search_results]

    return {
        'graph': {'nodes': nodes, 'edges': edges},
        'search_results': search_results,
        'sidebar': sidebar
    }

# --- Helper Functions for Streamlit UI ---

def show_graph(graph_data):
    """
    Displays the NetworkX graph using PyVis in Streamlit.
    """
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='remote', directed=False)
    net.force_atlas_2based()

    for node in graph_data['nodes']:
        color = "#FFD700" if node['type'] == 'entity' else "#6A5ACD"
        node_size = 20 if node['type'] == 'document' else 15
        net.add_node(node['id'], label=node['id'], title=f"Type: {node['type']}", color=color, size=node_size)

    for edge in graph_data['edges']:
        net.add_edge(edge['source'], edge['target'], title=edge['relation'], color="#CCCCCC", value=1)

    try:
        net.save_graph(GRAPH_HTML_PATH)
        with open(GRAPH_HTML_PATH, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=600, scrolling=True)
    except Exception as e:
        st.error(f"Failed to display graph: {e}")
        st.write("Please check if the PyVis HTML file can be written.")

@st.cache_data(show_spinner="Loading and processing data...")
def load_and_process_data(directory_path):
    """
    Loads text files, generates summaries, and builds the memory graph.
    Cached to avoid re-processing on every Streamlit rerun.
    """
    st.info(f"Ingesting data from: {directory_path}")
    ingested_data = ingest_text_files(directory_path)

    if not ingested_data:
        st.warning("No text files found or directory is invalid. Please check the path.")
        return {}, {}, nx.Graph(), [] # Also return empty entities list

    st.info("Generating summaries...")
    summaries = {filename: summarize_text(content, num_sentences=2)
                 for filename, content in ingested_data.items()}

    st.info("Building memory graph...")
    memory_graph = build_memory_graph(summaries)

    # Extract all unique entities for the "Key Entities" sidebar
    all_entities = sorted(list(set(n for n, data in memory_graph.nodes(data=True) if data.get('type') == 'entity')))


    st.success(f"Successfully processed {len(ingested_data)} documents.")
    return ingested_data, summaries, memory_graph, all_entities

# --- Streamlit Application ---

st.set_page_config(layout="wide", page_title="AI Memory Assistant")

st.title("AI Memory Assistant")

# --- Initialize session state variables ---
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
if 'ingested_data' not in st.session_state:
    st.session_state['ingested_data'] = {}
if 'summaries' not in st.session_state:
    st.session_state['summaries'] = {}
if 'memory_graph' not in st.session_state:
    st.session_state['memory_graph'] = nx.Graph()
if 'all_entities' not in st.session_state: # To store all unique entities
    st.session_state['all_entities'] = []
if 'search_results' not in st.session_state:
    st.session_state['search_results'] = []
if 'recent_searches' not in st.session_state: # To store a history of searches
    st.session_state['recent_searches'] = [] # Max length, e.g., 5-10 items

# --- Sidebar for Data Ingestion and Configuration ---
st.sidebar.header("Configuration")
directory_input = st.sidebar.text_input("Enter directory path for text files:", DEFAULT_DATA_DIRECTORY, key="dir_input")

if st.sidebar.button("Load/Reload Data", key="load_button") or not st.session_state['data_loaded']:
    # Only run data loading if button clicked OR if data isn't loaded yet (initial run)
    st.session_state['ingested_data'], st.session_state['summaries'], \
    st.session_state['memory_graph'], st.session_state['all_entities'] = \
        load_and_process_data(directory_input)
    if st.session_state['ingested_data']:
        st.session_state['data_loaded'] = True
    else:
        st.session_state['data_loaded'] = False

# --- Main Content Area ---

# Display tabs
tab1, tab2 = st.tabs(["Memory Graph & Search", "Document Summaries"])

with tab1:
    st.header("Memory Graph & Document Search")

    # Search bar at the top of the main area
    query = st.text_input("Enter your search query:", key="search_query_input")
    search_button_col, _ = st.columns([0.15, 0.85]) # Adjust width for button
    with search_button_col:
        search_triggered = st.button("Search", key="search_button")

    if search_triggered: # Trigger search on button click
        if query and st.session_state['ingested_data']:
            search_results = search_documents(query, st.session_state['ingested_data'], top_k=3)
            st.session_state['search_results'] = search_results

            # Add to recent searches (keep max 5 items)
            st.session_state['recent_searches'].insert(0, {
                'query': query,
                'time': datetime.datetime.now().strftime("%H:%M:%S"),
                'top_result': search_results[0]['document'] if search_results else "No Match"
            })
            st.session_state['recent_searches'] = st.session_state['recent_searches'][:5] # Keep only latest 5
        else:
            st.warning("Please load data first or enter a query.")
            st.session_state['search_results'] = [] # Clear results if query is empty or no data

    # Use columns for Response Panel and Graph Visualization
    response_col, graph_col = st.columns([0.4, 0.6]) # Adjust proportions as needed

    with response_col:
        st.subheader("Response Panel (Cited Answers)")
        if st.session_state.get('data_loaded') and st.session_state['ingested_data']:
            if st.session_state['search_results']:
                for i, result in enumerate(st.session_state['search_results']):
                    st.markdown(f"**{i+1}. Document:** `{result['document']}` (Similarity: `{result['similarity']:.4f}`)")
                    st.write(f"**Excerpt:** {result['excerpt']}")
                    st.markdown("---")
            elif search_triggered and not st.session_state['search_results']:
                st.info("No documents found matching your query.")
            else:
                st.info("Enter a query and click 'Search' to see results here.")
        else:
            st.info("Load data from the sidebar to enable search.")


    with graph_col:
        st.subheader("Memory Graph Visualization")
        if st.session_state.get('data_loaded') and st.session_state['memory_graph'].number_of_nodes() > 0:
            graph_obj = st.session_state['memory_graph']
            st.write(f"**Graph Info:** {graph_obj.number_of_nodes()} nodes, {graph_obj.number_of_edges()} edges")
            graph_data = prepare_ui_data(graph_obj, [])
            show_graph(graph_data['graph'])
        else:
            st.info("Load data from the sidebar to visualize the memory graph.")

with tab2:
    st.header("Document Summaries")
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
st.sidebar.subheader("Recent Searches")
if st.session_state['recent_searches']:
    for i, search_item in enumerate(st.session_state['recent_searches']):
        st.sidebar.markdown(f"**{i+1}. {search_item['time']}**")
        st.sidebar.markdown(f"Query: `{search_item['query']}`")
        st.sidebar.markdown(f"Top: `{search_item['top_result']}`")
        if i < len(st.session_state['recent_searches']) - 1:
            st.sidebar.markdown("---")
else:
    st.sidebar.info("No recent searches yet.")

st.sidebar.markdown("---")
st.sidebar.subheader("Key Entities (from all documents)")
if st.session_state.get('data_loaded') and st.session_state['all_entities']:
    # Display top 10 entities or all if fewer than 10
    entities_to_show = st.session_state['all_entities'][:10]
    for entity in entities_to_show:
        st.sidebar.markdown(f"- **{entity}**")
    if len(st.session_state['all_entities']) > 10:
        st.sidebar.markdown(f"*(and {len(st.session_state['all_entities']) - 10} more...)*")
else:
    st.sidebar.info("Load data to see entities.")
