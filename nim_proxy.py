from flask import Flask, request, jsonify, Response
import requests
import json
import time

app = Flask(__name__)

# Configuration
NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY = "your_nvidia_api_key_here"  # Replace with your NVIDIA API key

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.json
        
        # Extract OpenAI format parameters
        messages = data.get('messages', [])
        model = data.get('model', 'meta/llama-3.1-405b-instruct')
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 1024)
        stream = data.get('stream', False)
        
        # Prepare NVIDIA NIM request
        nim_payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        # Add optional parameters if present
        if 'top_p' in data:
            nim_payload['top_p'] = data['top_p']
        if 'frequency_penalty' in data:
            nim_payload['frequency_penalty'] = data['frequency_penalty']
        if 'presence_penalty' in data:
            nim_payload['presence_penalty'] = data['presence_penalty']
        
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        if stream:
            # Handle streaming response
            return handle_streaming_response(nim_payload, headers)
        else:
            # Handle non-streaming response
            response = requests.post(
                f"{NVIDIA_NIM_BASE_URL}/chat/completions",
                headers=headers,
                json=nim_payload,
                timeout=120
            )
            
            if response.status_code == 200:
                return jsonify(response.json())
            else:
                return jsonify({
                    "error": {
                        "message": f"NVIDIA NIM API error: {response.text}",
                        "type": "nvidia_error",
                        "code": response.status_code
                    }
                }), response.status_code
                
    except Exception as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "proxy_error"
            }
        }), 500

def handle_streaming_response(payload, headers):
    def generate():
        try:
            with requests.post(
                f"{NVIDIA_NIM_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=120
            ) as response:
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            yield line + '\n\n'
        except Exception as e:
            error_data = {
                "error": {
                    "message": str(e),
                    "type": "stream_error"
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models"""
    try:
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}"
        }
        
        response = requests.get(
            f"{NVIDIA_NIM_BASE_URL}/models",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            # Return a default model list if the endpoint fails
            return jsonify({
                "object": "list",
                "data": [
                    {
                        "id": "meta/llama-3.1-405b-instruct",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "nvidia"
                    },
                    {
                        "id": "meta/llama-3.1-70b-instruct",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "nvidia"
                    }
                ]
            })
    except Exception as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "proxy_error"
            }
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    # Run on all interfaces, port 8000
    # For production, use a proper WSGI server like gunicorn
    app.run(host='0.0.0.0', port=8000, debug=False)