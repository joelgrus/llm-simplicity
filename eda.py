import json
from dataclasses import dataclass
from typing import Optional
from uuid import UUID
from datetime import datetime
import glob

import pandas as pd

from conversations import Conversation, Role, ContentType

@dataclass
class Part:
    author: Role
    content_type: ContentType
    texts: list[str]
    model_slug: Optional[str] = None

    @property
    def speaker(self) -> str:
        if self.author.value == "assistant":
            return "ChatGPT"
        elif self.author.value == "user":
            return "me"
        else:
            return self.author.value

@dataclass
class Chat:
    id: UUID
    title: Optional[str]
    create_time: datetime
    parts: list[Part]

    @property
    def model_slug(self) -> Optional[str]:
        slugs = [part.model_slug for part in self.parts if part.model_slug]
        if slugs:
            return slugs[0]
        else:
            return None

    def to_dict(self) -> dict:
        return {
            'id': str(self.id),
            'title': self.title,
            'create_time': self.create_time.isoformat(),
            'parts': [
                {
                    'author': part.author.value,
                    'speaker': part.speaker,
                    'content_type': part.content_type.value,
                    'text': part.texts[0]
                }
                for part in self.parts
                if part.texts and part.texts[0]
            ],
            'model_slug': self.model_slug.value if self.model_slug else None
        }

    def to_markdown(self) -> str:
        return '\n\n'.join([
            f"# {self.title}"
        ] + [
            f"**{part.author.value}**: {part.texts[0]}"
            for part in self.parts
            if part.texts[0]
        ])

    def text(self) -> str:
        print("\n\n".join([
            f"{part.speaker}: {part.texts[0]}"
            for part in self.parts
            if part.texts[0]
        ]))

def conversation_to_chat(convo: Conversation) -> Chat:
    id = convo.id
    title = convo.title
    create_time = datetime.fromtimestamp(convo.create_time)
    parts = []

    for mapping in convo.mapping.values():
        if (message := mapping.message):
            role = message.author.role
            content = message.content
            content_type = content.content_type
            metadata = mapping.message.metadata 
            model_slug = metadata.model_slug if metadata else None

            if content.parts:
                parts.append(Part(role, content_type, content.parts, model_slug))
            elif content.text:
                parts.append(Part(role, content_type, [content.text], model_slug))
            elif content.result:
                parts.append(Part(role, content_type, [content.result], model_slug))
            else:
                raise ValueError(f"Unknown content type: {content}")
    
    return Chat(id, title, create_time, parts)



with open('data/conversations.json') as f:
    data = json.load(f)

convos = [Conversation.from_dict(c) for c in data]
convos.sort(key=lambda c: c.create_time)

chats = [conversation_to_chat(c) for c in convos]

df = pd.DataFrame([chat.to_dict() for chat in chats])
df['created_at'] = pd.to_datetime(df['create_time'])


class Result(dict):
    @property
    def chat(self):
        return self["chat"]
    
    @property
    def analysis(self):
        return self["analysis"]
    
    @property
    def arguments(self):
        try:
            return json.loads(self.analysis['choices'][0]['message']['function_call']['arguments'])
        except:
            return None

results = []

for fn in glob.glob('output/*.json'):
    with open(fn) as f:
        try:
            results.append(Result(json.load(f)))
        except:
            print(f"Error loading {fn}")