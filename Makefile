# Persona-Lens Pipeline Makefile
# This Makefile connects all components of the Persona-Lens pipeline

# Define common variables
PYTHON = python3
PIP = pip3
DATA_DIR = Dataset
EMBEDDINGS_DIR = embeddings
PLOTS_DIR = plots
GITHUB_USERS_DIR = github-users

# Default target
.PHONY: all
all: setup install scrape extract chunk embed query

# Setup directories
.PHONY: setup
setup:
	@echo "Setting up required directories..."
	@mkdir -p $(DATA_DIR) $(EMBEDDINGS_DIR) $(PLOTS_DIR) $(GITHUB_USERS_DIR)

# Install dependencies
.PHONY: install
install:
	@echo "Installing required dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	# Install faiss-cpu specifically since it's critical for the embedding functionality
	$(PIP) install faiss-cpu

# Step 1: Scrape GitHub data
.PHONY: scrape
scrape: setup install
	@echo "Scraping GitHub user data..."
	$(PYTHON) github_scraper.py

# Step 2: Extract features from the raw data
.PHONY: extract
extract: setup install
	@echo "Extracting features from GitHub user data..."
	$(PYTHON) feature_extractor.py

# Step 3: Create semantic chunks from the processed data
.PHONY: chunk
chunk: setup install
	@echo "Creating semantic chunks from processed data..."
	$(PYTHON) data_chunking.py

# Step 4: Create vector embeddings for the chunks
.PHONY: embed
embed: setup install
	@echo "Creating vector embeddings for semantic chunks..."
	$(PYTHON) embedding_indexer.py

# Step 5: Test queries against the embedded chunks
.PHONY: query
query: setup install
	@echo "Starting interactive query testing..."
	$(PYTHON) query_tester.py

# Step 6: LLM interface for natural language queries
.PHONY: llm
llm: setup install embed
	@echo "Starting LLM interface for natural language queries..."
	$(PYTHON) llm_interface.py

# Step 7: Run the LLM prompt connector for integration with ChatGPT/Gemini
.PHONY: llm-connector
llm-connector: setup install embed
	@echo "Starting LLM prompt connector for ChatGPT/Gemini integration..."
	$(PYTHON) llm_prompt_connector.py

# Step 8: Run the API service for integration with external LLMs
.PHONY: api
api: setup install embed
	@echo "Starting API service for external LLM integration..."
	$(PIP) install flask
	$(PYTHON) api_service.py

# Step 9: Run the Gemini client that connects to the API service (Python 3.6 compatible)
.PHONY: gemini
gemini: setup install
	@echo "Starting Gemini client for Persona-Lens (Python 3.6 compatible)..."
	@echo "Please make sure to set your Gemini API key in the .env file:"
	@echo "GEMINI_API_KEY=your_api_key_here"
	$(PYTHON) gemini_client.py

# Run the full pipeline from start to finish
.PHONY: pipeline
pipeline: setup install scrape extract chunk embed

# Run the full pipeline including LLM interface
.PHONY: pipeline-llm
pipeline-llm: pipeline llm

# Run the full pipeline including ChatGPT/Gemini connector
.PHONY: pipeline-connector
pipeline-connector: pipeline llm-connector

# Run the full pipeline including API service
.PHONY: pipeline-api
pipeline-api: pipeline api

# Run both API service and Gemini client (in separate terminals)
.PHONY: pipeline-gemini
pipeline-gemini: pipeline
	@echo "Starting API service and Gemini client..."
	@echo "Please run these commands in separate terminals:"
	@echo "Terminal 1: make api"
	@echo "Terminal 2: make gemini"
	@echo "Make sure to set your Gemini API key in the .env file:"
	@echo "GEMINI_API_KEY=your_api_key_here"

# Clean generated files (use with caution)
.PHONY: clean
clean:
	@echo "Cleaning generated files..."
	rm -f semantic_chunks.json
	rm -f $(EMBEDDINGS_DIR)/*.faiss $(EMBEDDINGS_DIR)/*.pkl

# Clean everything (use with extreme caution)
.PHONY: clean-all
clean-all: clean
	@echo "Cleaning all generated data..."
	rm -f $(DATA_DIR)/*.csv
	rm -f $(PLOTS_DIR)/*.png

# Check dependencies
.PHONY: check-deps
check-deps:
	@echo "Checking for required Python packages..."
	@$(PYTHON) -c "import faiss" 2>/dev/null || (echo "Package 'faiss' is not installed. Run 'make install' first." && exit 1)
	@$(PYTHON) -c "import pandas" 2>/dev/null || (echo "Package 'pandas' is not installed. Run 'make install' first." && exit 1)
	@$(PYTHON) -c "import torch" 2>/dev/null || (echo "Package 'torch' is not installed. Run 'make install' first." && exit 1)
	@$(PYTHON) -c "import sentence_transformers" 2>/dev/null || (echo "Package 'sentence-transformers' is not installed. Run 'make install' first." && exit 1)
	@echo "All required packages are installed."

# Help target
.PHONY: help
help:
	@echo "Persona-Lens Pipeline"
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@echo "  all            Run the entire pipeline (default)"
	@echo "  setup          Create necessary directories"
	@echo "  install        Install required dependencies"
	@echo "  check-deps     Check if all required dependencies are installed"
	@echo "  scrape         Run GitHub scraper to collect user data"
	@echo "  extract        Run feature extraction on collected data"
	@echo "  chunk          Create semantic chunks from processed data"
	@echo "  embed          Create vector embeddings for semantic chunks"
	@echo "  query          Start interactive query interface"
	@echo "  llm            Start LLM interface for natural language queries"
	@echo "  llm-connector  Start LLM prompt connector for ChatGPT/Gemini integration"
	@echo "  api            Start API service for external LLM integration"
	@echo "  gemini         Start Gemini client for Persona-Lens (Python 3.6 compatible)"
	@echo "  pipeline       Run the complete pipeline (except querying)"
	@echo "  pipeline-llm   Run complete pipeline and launch LLM interface"
	@echo "  pipeline-connector Run complete pipeline and launch ChatGPT/Gemini connector"
	@echo "  pipeline-api   Run complete pipeline and launch API service"
	@echo "  pipeline-gemini Run pipeline and provide instructions for API+Gemini setup"
	@echo "  clean          Remove generated index files"
	@echo "  clean-all      Remove all generated data (use with caution)"
	@echo "  help           Show this help message"