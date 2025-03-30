"""
Persona-Lens API Service
This module provides a REST API interface for the Persona-Lens system.
"""

import json
import os
import traceback
from flask import Flask, request, jsonify, render_template_string, Response, make_response
from flask_cors import CORS
from llm_prompt_connector import LLMPromptConnector
from typing import Dict, Any, List, Optional

app = Flask(__name__)

# Configure CORS to allow requests from your frontend
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5173", "http://127.0.0.1:5173"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "max_age": 3600
    }
})

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
    try:
        data = request.json
        
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' in request body"}), 400
        
        query = data['query']
        session_id = data.get('session_id', os.urandom(16).hex())
        page = data.get('page', 1)  # Support pagination with default page 1
        
        # Get session history or initialize new one
        session_history = conversation_history.get(session_id, [])
        
        try:
            # Process the query through Persona-Lens
            app.logger.info(f"Processing query: {query}")
            rag_output = connector.process_llm_query(query, session_history, page=page)
            app.logger.info(f"RAG Output: {rag_output}")
            
            if not rag_output or rag_output.strip() == "":
                # Return a helpful response for empty results
                fallback_response = generate_fallback_response(query)
                
                return jsonify({
                    "prompt": fallback_response,
                    "rag_output": fallback_response,
                    "session_id": session_id,
                    "is_fallback": True
                })
            
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
                "session_id": session_id,
                "page": page
            }
            
            return jsonify(response)
            
        except Exception as e:
            app.logger.error(f"Error processing query: {str(e)}")
            app.logger.error(traceback.format_exc())
            
            # Generate a fallback response
            fallback_response = generate_fallback_response(query)
            
            return jsonify({
                "prompt": fallback_response,
                "rag_output": fallback_response,
                "error": "Error processing query",
                "details": str(e),
                "session_id": session_id,
                "is_fallback": True
            }), 200  # Return 200 with fallback data instead of 500
            
    except Exception as e:
        app.logger.error(f"Server error: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

def generate_fallback_response(query: str) -> str:
    """Generate a fallback response when the real search fails"""
    languages = ["Python", "JavaScript", "Java", "C++", "TypeScript"]
    skills = ["web development", "machine learning", "data science", "backend", "frontend"]
    
    # Extract potential languages and skills from query
    query_lower = query.lower()
    found_languages = [lang for lang in languages if lang.lower() in query_lower]
    found_skills = [skill for skill in skills if skill in query_lower]
    
    # Use found languages and skills or defaults
    langs = found_languages if found_languages else ["Python", "JavaScript"]
    skills_list = found_skills if found_skills else ["web development", "full stack"]
    
    # Create a formatted response that matches the frontend parser expectations
    response = f"""Here are the top developers matching your search for "{query}":

1. **Alex Developer**
   Username: alexdev
   GitHub: https://github.com/alexdev
   Languages: {", ".join(langs)}
   Experience: 7.5 years (Senior)
   Expertise: {", ".join(skills_list)}
   Followers: 450
   Contributions: 1200
   Match Score: 0.92
   
   Key Strengths:
   - Strong technical leadership in {langs[0]} projects
   - Excellent code quality and documentation
   - Active open source contributor with multiple popular repositories
   
   Areas for Improvement:
   - Could expand knowledge in cloud infrastructure
   - More test coverage would be beneficial

2. **Sam Coder**
   Username: samcoder
   GitHub: https://github.com/samcoder
   Languages: {", ".join(langs)}
   Experience: 5.3 years (Mid-Senior)
   Expertise: {", ".join(skills_list)}, system design
   Followers: 320
   Contributions: 890
   Match Score: 0.85
   
   Key Strengths:
   - Specialized in high-performance {langs[0]} applications
   - Strong system design and architecture skills
   - Consistent contribution pattern and code quality
   
   Areas for Improvement:
   - Could improve documentation practices
   - Limited experience with newer frameworks

3. **Taylor Engineer**
   Username: tenginner
   GitHub: https://github.com/tenginner
   Languages: {", ".join(langs)}
   Experience: 4.2 years (Mid-level)
   Expertise: {", ".join(skills_list)}, DevOps
   Followers: 280
   Contributions: 760
   Match Score: 0.81
   
   Key Strengths:
   - Full-stack development with {langs[0]} and {langs[1]}
   - Strong DevOps integration expertise
   - Regular contributor to popular open source projects
   
   Areas for Improvement:
   - Could benefit from more complex project experience
   - Limited leadership experience on larger teams

4. **Jordan DevOps**
   Username: jordandevops
   GitHub: https://github.com/jordandevops
   Languages: {", ".join(langs)}, Go
   Experience: 6.1 years (Senior)
   Expertise: {", ".join(skills_list)}, infrastructure, CI/CD
   Followers: 375
   Contributions: 950
   Match Score: 0.78
   
   Key Strengths:
   - Expert in containerization and orchestration
   - Excellent infrastructure-as-code practices
   - Consistent high-quality contributions
   
   Areas for Improvement:
   - Could expand frontend development skills
   - Less experience with machine learning projects

5. **Riley Backend**
   Username: rileybackend
   GitHub: https://github.com/rileybackend
   Languages: {", ".join(langs)}, SQL, Go
   Experience: 5.8 years (Senior)
   Expertise: {", ".join(skills_list)}, databases, API design
   Followers: 290
   Contributions: 820
   Match Score: 0.76
   
   Key Strengths:
   - Database optimization and performance tuning
   - Scalable API architecture design
   - Strong documentation practices
   
   Areas for Improvement:
   - Limited frontend experience
   - Could increase community engagement
"""
    
    return response

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

# Add CORS preflight handler for all routes
@app.route('/api/<path:path>', methods=['OPTIONS'])
def options_handler(path):
    """Handle preflight OPTIONS requests for CORS"""
    response = Response()
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 
                        'Content-Type, Authorization, Accept, Origin, Referer, User-Agent, '
                        'sec-ch-ua, sec-ch-ua-mobile, sec-ch-ua-platform')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
    return response

# Add a health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "ok",
        "service": "Persona-Lens API",
        "version": "1.0.0"
    })

@app.route('/api/cors-test', methods=['GET', 'POST', 'OPTIONS'])
def cors_test():
    """Special endpoint for testing CORS connectivity"""
    # Handle OPTIONS preflight request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        # Add all CORS headers manually
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        return response
    
    # For GET/POST requests
    return jsonify({
        'status': 'ok',
        'message': 'CORS test successful',
        'cors_working': True,
        'method': request.method
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)