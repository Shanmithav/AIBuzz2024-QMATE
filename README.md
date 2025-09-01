Team Name: QueryMates

Idea Title: QMATE (Don't worry buddy I got it all)

Idea url: 

Team Members: Shanmitha Vangeti, Pruthvi Raj Singh

Programming language used: python

AI Hub Model links: 

Target device: PC

Implementation Summary
facebook/bart-large-cnn, all-MiniLM-L6-v2, dbmdz/bert-large-cased-finetuned-conll03-eng

The app uses the Snapdragon® Neural Processing Unit (NPU) to run the LLM model for acceleration on Edge PC/laptop device.

In the current prototype version we have not included the real ai-models , we have some limitations in accessing the QNN models, created a dummy or a mockup version of the prototype, will share the working one by the final submission. 
Target device:

Programming Language Used: Python
AI Hub Model Links:
No specific pre-trained AI "models" from a hub (like Hugging Face, TensorFlow Hub, etc.) are directly linked or imported.
The project primarily uses traditional NLP techniques for text processing, summarization, and similarity:
nltk (Natural Language Toolkit) for tokenization (sentences) and stopwords.
scikit-learn (sklearn.feature_extraction.text.TfidfVectorizer, sklearn.metrics.pairwise.cosine_similarity) for TF-IDF vectorization and cosine similarity, which are used for summarization and document search.
networkx for graph data structure.
pyvis for interactive graph visualization.
Target Device: 

PC: Primary target, as it's a Streamlit web application typically run in a browser on a desktop/laptop.

Implementation Summary: The AI Memory Assistant is a Streamlit web application designed to help users ingest, summarize, and search through collections of text documents (specifically .txt files). It builds a "memory graph" to visualize relationships between documents and extracted key entities.

Key Components:

Data Ingestion: Reads all .txt files from a specified directory.
Summarization: Uses TF-IDF to extract the most important sentences from each document, providing a concise summary.
Memory Graph Construction: Builds a NetworkX graph where nodes represent documents and extracted "entities" (capitalized words/phrases). Edges indicate that a document "mentions" an entity.
Document Search: Performs semantic search using TF-IDF and cosine similarity to find documents most relevant to a user's query. It returns top k results with similarity scores and excerpts.
Interactive Visualization: Uses PyVis to render the memory graph interactively within the Streamlit interface.
Streamlit UI: Provides an intuitive web interface with:
Configuration sidebar for data directory input and reloading.
Tabs for "Memory Graph & Search" and "Document Summaries."
Search bar, response panel for search results, and the graph visualization.
A dedicated tab to view full document summaries and original text.
Sidebar features for "Recent Searches" and listing "Key Entities."
Installation & Setup Steps:

Prerequisites:

Python 3.7+ (recommended 3.9+)
Clone/Download the project:

Place the AI_project.py file in your desired project directory.
Install Python dependencies: Open your terminal or command prompt, navigate to the directory where AI_project.py is located, and run:
pip install streamlit networkx scikit-learn pyvis nltk
Download NLTK data: The script attempts to download punkt and stopwords automatically if not found. However, if you encounter issues, you can pre-download them by opening a Python interpreter and running:
import nltk
nltk.download('punkt')
nltk.download('stopwords')
Prepare your data directory:
Create a directory containing your .txt files. For example, create a folder named sample_files and place some text files (e.g., doc1.txt, doc2.txt) inside it.
Important: Update the DEFAULT_DATA_DIRECTORY variable in AI_project.py to point to your actual directory.
DEFAULT_DATA_DIRECTORY = r"C:\path\to\your\sample_files"
(Ensure to use forward slashes / or double backslashes \\ for Windows paths, or an r prefix for raw string literals).
Expected Output / Behavior:

Initial Load: Upon running, the Streamlit app will open in your web browser. It will initially show a warning if no data is loaded.
Data Loading: Use the "Configuration" sidebar:
Enter the path to your text files in the "Enter directory path for text files:" input box.
Click "Load/Reload Data".
You'll see "Loading and processing data..." spinner.
Success messages will appear: "Ingesting data...", "Generating summaries...", "Building memory graph...", "Successfully processed X documents."
Memory Graph Tab:
The left panel ("Response Panel") will initially be empty, prompting for a search.
The right panel ("Memory Graph Visualization") will display an interactive graph with purple nodes for documents and yellow nodes for entities, connected by grey edges. You can drag nodes, zoom, and hover over them for details.
Graph info (nodes/edges count) will be displayed.
Search Functionality:
Enter a query in the "Enter your search query:" box (e.g., "AI development" or "key concepts").
Click the "Search" button.
The "Response Panel" will populate with the top 3 most similar documents, showing their names, similarity scores, and short excerpts.
The "Recent Searches" section in the sidebar will update with your query and top result.
Document Summaries Tab:
Select a document from the dropdown list.
Its generated summary will be displayed.
An "Expander" allows you to view the original full text of the selected document.
Sidebar Information:
"Recent Searches" will list your last few queries.
"Key Entities" will display prominent entities extracted from all documents.
Any Additional Steps Required for Running:

Ensure your DEFAULT_DATA_DIRECTORY in AI_project.py is correctly set and contains .txt files.
To run the Streamlit application, open your terminal or command prompt, navigate to the directory where AI_project.py is saved, and execute:
streamlit run AI_project.py
This command will typically open the application in your default web browser (usually at http://localhost:8501).
Submission Checklist (based on standard project requirements):
output snippets:
<img width="1245" height="550" alt="image" src="https://github.com/user-attachments/assets/1c979958-a46e-44e9-b034-0eb0e7f04a1d" />
<img width="1259" height="671" alt="image" src="https://github.com/user-attachments/assets/21f5ad61-7381-404d-9d54-c7251a263521" />


