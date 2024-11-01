from fastapi import WebSocket
from typing import List
import json
from lib.providers import get_chat_model, get_embeddings_model
from agents.web_search_agent import handle_web_search

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

    async def handle_connection(self, websocket: WebSocket):
        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)

                if message["type"] == "web_search":
                    llm = get_chat_model()
                    embeddings = get_embeddings_model()
                    result = await handle_web_search(message["query"], message["history"], llm, embeddings)
                    await self.send_personal_message(json.dumps(result), websocket)
                else:
                    await self.send_personal_message(json.dumps({"error": "Unknown message type"}), websocket)
        except Exception as e:
            print(f"Error handling WebSocket connection: {str(e)}")
            await self.disconnect(websocket)
