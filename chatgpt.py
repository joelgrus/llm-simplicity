import json
import os

import openai

from eda import Chat

MODEL = "gpt-4-0613"

PROMPT = """I am going to give you the transcript of an OpenAI chat.
I want you to provide for me a few things:

1. a brief summary of the chat (say, 1-2 sentences)
2. a list of keywords that apply to the chat
3. a brief topic description of the chat (a few words)
4. a rating from 0 to 100 of how *interesting* the chat is
5. a rating from 0 to 100 of how well ChatGPT did on the chat
6. a rating from 0 to 100 of how much the chat is related to software engineering
7. a rating from 0 to 100 of how much the chat is related to data science
8. a rating from 0 to 100 of how offensive the chat might be to a reader
9. a rating from 0 to 100 of how embarrassing the chat might be for the chatter

Here is the chat, in markdown format:

{0}
"""

def analyze_chat(text: str) -> dict:    
    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT.format(text)
                }
            ],
            functions=[{
                "name": "stats",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "A brief summary of the chat (say, 1-2 sentences)"
                        },
                        "keywords": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "A list of keywords that apply to the chat"
                        },
                        "topic": {
                            "type": "string",
                            "description": "A brief topic description of the chat (a few words)"
                        },
                        "interesting": {
                            "type": "number",
                            "description": "A rating from 0 to 100 of how *interesting* the chat is"
                        },
                        "quality": {
                            "type": "number",
                            "description": "A rating from 0 to 100 of how well ChatGPT did on the chat"
                        },                
                        "softwareEngineering": {
                            "type": "number",
                            "description": "A rating from 0 to 100 of how much the chat is related to software engineering"
                        },                
                        "dataScience": {
                            "type": "number",
                            "description": "A rating from 0 to 100 of how much the chat is related to data science"
                        },                
                        "offensiveness": {
                            "type": "number",
                            "description": "A rating from 0 to 100 of how offensive the chat might be to a reader"
                        },                
                        "embarrassment": {
                            "type": "number",
                            "description": "A rating from 0 to 100 of how embarrassing the chat might be for the chatter"
                        }                
                    }
                }
            }]
        )
    
        #return json.loads(response['choices'][0]['message']['function_call']['arguments'])
        return response.to_dict_recursive()

    except openai.error.InvalidRequestError as e:
        message = getattr(e, "_message", "")
        if "maximum context length" in message:
            return {}
        else:
            raise e


def do_analysis(chat: Chat):
    fn = f'output/{chat.id}.json'

    if os.path.exists(fn):
        print(f"Skipping {chat.id}, file already exists")
        return

    output = analyze_chat(chat.to_markdown())
    blob = { "chat": chat.to_dict(), "analysis": output }

    with open(f'output/{chat.id}.json', 'w') as f:
        json.dump(blob, f, indent=2)