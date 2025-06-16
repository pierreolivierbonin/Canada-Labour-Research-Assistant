import requests
import json

# Generic OpenAI-compatible API functions
def oai_compatible_request(api_url, headers, data):
    """Generic non-streaming OpenAI-compatible API request"""
    response = requests.post(api_url, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Error in API call: {response.status_code} - {response.text}")

def oai_compatible_request_stream(api_url, headers, data):
    """Generic streaming OpenAI-compatible API request"""
    response = requests.post(api_url, headers=headers, json=data, stream=True)
    
    if response.status_code != 200:
        raise Exception(f"Error in streaming API call: {response.status_code} - {response.text}")

    # Create a byte buffer to collect incoming data
    byte_buffer = b""

    # iterate through the incoming bytes
    for byte_data in response.iter_content():
        byte_buffer += byte_data

        # Check for the end of a completed chunk
        while b'}\n\n' in byte_buffer:
            # Extract the next full completed chunk from the byte buffer
            chunk_end_idx = byte_buffer.index(b'}\n\n') + 3  # +3 to include "}\n\n"
            completed_chunk = byte_buffer[:chunk_end_idx].decode('utf-8')

            # Remove "data: " prefix and decode the JSON
            json_str = completed_chunk.replace("data: ", "", 1)

            # Remove anything before the first "{", without using split (use index instead)
            if "{" in json_str:
                json_str = json_str[json_str.index("{"):]

            try:
                chunk = json.loads(json_str)
                byte_buffer = byte_buffer[chunk_end_idx:]

                # Extract the message
                choices = chunk.get('choices', [])
                if choices:
                    chunk_message = choices[0].get('delta', {})
                    content = chunk_message.get("content", "")
                    if content:
                        yield content
            except json.JSONDecodeError:
                # Skip malformed JSON chunks
                byte_buffer = byte_buffer[chunk_end_idx:]
                continue

            # Check for the [DONE] marker
            if b"data: [DONE]\n\n" in byte_buffer:
                byte_buffer = b""
                break