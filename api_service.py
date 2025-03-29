"""
Persona-Lens API Service
This module provides a REST API interface for the Persona-Lens system, 
allowing external LLMs like Gemini or ChatGPT to query the system via API calls.
"""

import json
import os
from flask import Flask, request, jsonify, render_template_string
from llm_prompt_connector import LLMPromptConnector
from typing import Dict, Any, List, Optional

app = Flask(__name__)

# Initialize the LLM prompt connector
connector = LLMPromptConnector()

# Define the base prompt template
BASE_PROMPT_TEMPLATE = """
You are an AI recruiting assistant powered by Persona-Lens, a sophisticated developer talent search platform. You have access to a database of developer profiles with information about their skills, experience, projects, and other metrics. Your task is to help recruiters find the best candidates for their needs.

When processing a recruiter query, follow these steps:
1. Understand the intent (finding candidates with specific skills, experience levels, etc.)
2. Search the developer database using the Persona-Lens RAG pipeline
3. Format the results in a clear, structured way
4. For follow-up queries about specific candidates, provide more detailed profile information

The Persona-Lens system can:
- Find candidates based on programming languages (Python, JavaScript, Java, etc.)
- Filter by experience level (junior/beginner, mid-level, senior/expert)
- Rank candidates by relevance, popularity, or other metrics
- Provide detailed profiles with GitHub metrics, projects, and skills

Based on your query, here are the matching candidates from the Persona-Lens system:

{rag_output}

Maintain a professional, recruiter-friendly tone and format results in a way that makes it easy to compare candidates.
You can ask for more details about any specific candidate mentioned above.
"""

# Store conversation history for each session
conversation_history: Dict[str, List[Dict[str, str]]] = {}

# HTML template for the root route
ROOT_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Persona-Lens API Service</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        h2 {
            color: #3498db;
            margin-top: 30px;
        }
        code {
            background-color: #f8f8f8;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: Consolas, Monaco, monospace;
            font-size: 0.9em;
        }
        pre {
            background-color: #f8f8f8;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        .endpoint {
            margin-bottom: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }
        .method {
            font-weight: bold;
            color: #e74c3c;
        }
        .url {
            color: #2980b9;
        }
    </style>
</head>
<body>
    <h1>Persona-Lens API Service</h1>
    <p>Welcome to the Persona-Lens API Service. This API allows you to search for developer candidates and get detailed information about specific developers.</p>
    
    <h2>Available Endpoints</h2>
    
    <div class="endpoint">
        <h3>Search for Candidates</h3>
        <p><span class="method">POST</span> <span class="url">/api/persona-lens/query</span></p>
        <p>Search for developer candidates based on your query.</p>
        <h4>Request Body:</h4>
        <pre>
{
  "query": "give me top 10 candidates good at Python",
  "session_id": "optional-session-id"
}
        </pre>
        <h4>Example Usage:</h4>
        <pre>
curl -X POST http://localhost:5000/api/persona-lens/query \\
  -H "Content-Type: application/json" \\
  -d '{"query": "give me top 5 candidates good at Python"}'
        </pre>
    </div>
    
    <div class="endpoint">
        <h3>Get Detailed Profile</h3>
        <p><span class="method">POST</span> <span class="url">/api/persona-lens/detailed-profile</span></p>
        <p>Get detailed information about a specific developer.</p>
        <h4>Request Body:</h4>
        <pre>
{
  "username": "username-of-candidate",
  "session_id": "optional-session-id"
}
        </pre>
        <h4>Example Usage:</h4>
        <pre>
curl -X POST http://localhost:5000/api/persona-lens/detailed-profile \\
  -H "Content-Type: application/json" \\
  -d '{"username": "AllenDowney"}'
        </pre>
    </div>
    
    <h2>Integration with Gemini/ChatGPT</h2>
    <p>This API is designed to be used with LLMs like Gemini or ChatGPT. The responses include formatted prompts that can be directly sent to these LLMs.</p>
    
    <h2>Status</h2>
    <p>API Status: <strong>Running</strong></p>
    <p>Loaded Profiles: <strong>{{ profile_count }}</strong></p>
    <p>Current Time: <strong>{{ current_time }}</strong></p>
    
    <footer style="margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.8em;">
        <p>Persona-Lens API Service | &copy; {{ current_year }} Persona-Lens</p>
    </footer>
</body>
</html>
"""

@app.route('/')
def root():
    """
    Root route that shows API documentation
    """
    import datetime
    
    # Get the number of profiles loaded (if available)
    profile_count = "N/A"
    if hasattr(connector, 'user_data') and hasattr(connector.user_data, 'shape'):
        profile_count = f"{connector.user_data.shape[0]:,}"
    
    # Get current time
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    current_year = now.year
    
    # Render the HTML template
    return render_template_string(
        ROOT_HTML_TEMPLATE, 
        profile_count=profile_count,
        current_time=current_time,
        current_year=current_year
    )

@app.route('/api/persona-lens/query', methods=['POST'])
def query_persona_lens():
    """
    API endpoint for querying the Persona-Lens system
    
    Request body:
    {
        "query": "give me top 10 candidates good at Python",
        "session_id": "unique-session-identifier"  # Optional
    }
    
    Response:
    {
        "prompt": "Complete prompt with RAG output incorporated",
        "rag_output": "Raw RAG output from Persona-Lens",
        "session_id": "Session ID for maintaining conversation history"
    }
    """
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400
    
    query = data['query']
    session_id = data.get('session_id', os.urandom(16).hex())  # Generate session ID if not provided
    
    # Get session history or initialize new one
    session_history = conversation_history.get(session_id, [])
    
    # Process the query through Persona-Lens
    rag_output = connector.process_llm_query(query, session_history)
    
    # Create the complete prompt by combining template with RAG output
    complete_prompt = BASE_PROMPT_TEMPLATE.format(rag_output=rag_output)
    
    # Update conversation history
    session_history.append({"role": "user", "content": query})
    session_history.append({"role": "assistant", "content": rag_output})
    conversation_history[session_id] = session_history
    
    # Return complete prompt and raw RAG output
    response = {
        "prompt": complete_prompt,
        "rag_output": rag_output,
        "session_id": session_id
    }
    
    return jsonify(response)

@app.route('/api/persona-lens/detailed-profile', methods=['POST'])
def get_detailed_profile():
    """
    API endpoint for getting detailed information about a specific candidate
    
    Request body:
    {
        "username": "username-of-candidate",
        "session_id": "unique-session-identifier"  # Optional
    }
    
    Response:
    {
        "prompt": "Complete prompt with detailed profile",
        "profile_data": "Formatted detailed profile",
        "session_id": "Session ID for maintaining conversation history"
    }
    """
    data = request.json
    
    if not data or 'username' not in data:
        return jsonify({"error": "Missing 'username' in request body"}), 400
    
    username = data['username']
    session_id = data.get('session_id', os.urandom(16).hex())  # Generate session ID if not provided
    
    # Get detailed profile
    profile = connector.get_candidate_detailed_profile(username)
    formatted_profile = connector.format_detailed_profile(profile)
    
    # Create the complete prompt by combining template with profile data
    profile_prompt = f"""
You are an AI recruiting assistant powered by Persona-Lens, a sophisticated developer talent search platform.
A recruiter has asked for detailed information about a specific developer: {username}.
Here is the comprehensive profile information from Persona-Lens:

{formatted_profile}

Provide a helpful analysis of this candidate, highlighting their strengths and key areas of expertise.
    """
    
    # Get session history or initialize new one
    session_history = conversation_history.get(session_id, [])
    
    # Update conversation history
    query = f"Tell me more about {username}"
    session_history.append({"role": "user", "content": query})
    session_history.append({"role": "assistant", "content": formatted_profile})
    conversation_history[session_id] = session_history
    
    # Return complete prompt and profile data
    response = {
        "prompt": profile_prompt,
        "profile_data": formatted_profile,
        "session_id": session_id
    }
    
    return jsonify(response)

# Add a health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "ok",
        "service": "Persona-Lens API",
        "version": "1.0.0"
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)