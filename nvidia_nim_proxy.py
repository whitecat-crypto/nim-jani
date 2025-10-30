from flask import Flask, request, jsonify, Response
import requests
import json
import time

app = Flask(__name__)

# NVIDIA NIM Configuration
NVIDIA_API_KEY = "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"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "deepseek-ai/deepseek-r1-0528"

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models in OpenAI format"""
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "deepseek-r1",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "nvidia",
                "permission": [],
                "root": "deepseek-r1",
                "parent": None
            }
        ]
    })

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """Proxy chat completions from OpenAI format to NVIDIA NIM"""
    try:
        data = request.get_json()
        
        # Extract parameters from OpenAI format
        messages = data.get('messages', [])
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 1024)
        stream = data.get('stream', False)
        top_p = data.get('top_p', 1.0)
        
        # Build NVIDIA NIM request
        nvidia_payload = {
            "model": NVIDIA_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream
        }
        
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Make request to NVIDIA NIM
        nvidia_response = requests.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            headers=headers,
            json=nvidia_payload,
            stream=stream
        )
        
        if stream:
            # Handle streaming response
            def generate():
                for line in nvidia_response.iter_lines():
                    if line:
                        yield line + b'\n'
            
            return Response(generate(), content_type='text/event-stream')
        else:
            # Handle non-streaming response
            return jsonify(nvidia_response.json()), nvidia_response.status_code
            
    except Exception as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "proxy_error",
                "code": "internal_error"
            }
        }), 500

@app.route('/v1/completions', methods=['POST'])
def completions():
    """Legacy completions endpoint (converts to chat format)"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        # Convert to chat format
        messages = [{"role": "user", "content": prompt}]
        
        # Build request
        nvidia_payload = {
            "model": NVIDIA_MODEL,
            "messages": messages,
            "temperature": data.get('temperature', 0.7),
            "max_tokens": data.get('max_tokens', 1024),
            "top_p": data.get('top_p', 1.0),
            "stream": data.get('stream', False)
        }
        
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        nvidia_response = requests.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            headers=headers,
            json=nvidia_payload
        )
        
        # Convert response back to completions format
        chat_response = nvidia_response.json()
        if 'choices' in chat_response and len(chat_response['choices']) > 0:
            completion_response = {
                "id": chat_response.get('id', ''),
                "object": "text_completion",
                "created": chat_response.get('created', int(time.time())),
                "model": "deepseek-r1",
                "choices": [{
                    "text": chat_response['choices'][0]['message']['content'],
                    "index": 0,
                    "finish_reason": chat_response['choices'][0].get('finish_reason', 'stop')
                }],
                "usage": chat_response.get('usage', {})
            }
            return jsonify(completion_response)
        
        return jsonify(chat_response), nvidia_response.status_code
            
    except Exception as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "proxy_error",
                "code": "internal_error"
            }
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "service": "nvidia-nim-proxy"})

if __name__ == '__main__':
    print("=" * 60)
    print("NVIDIA NIM to OpenAI API Proxy Server")
    print("=" * 60)
    print(f"Model: {NVIDIA_MODEL}")
    print(f"Base URL: {NVIDIA_BASE_URL}")
    print("\nServer starting on http://0.0.0.0:5000")
    print("\nUsage in Janitor AI:")
    print("  API Endpoint: http://YOUR_SERVER_IP:5000/v1")
    print("  API Key: any-key (not validated)")
    print("  Model: deepseek-r1")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False)