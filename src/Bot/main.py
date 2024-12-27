from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, Security
from utils.llm_interface import LLM

app = FastAPI()
chatbot = LLM()

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    # TODO: check for api key and security
    await websocket.accept()
    conversation_history = []

    try:
        while True:
            # Receive user's message
            prompt = await websocket.receive_text()
            conversation_history.append(prompt)

            # Get response from the chatbot
            response = chatbot.generate_answer(prompt, user_id, conversation_history=conversation_history)
            conversation_history.append(response)

            # Send the response back to the client
            await websocket.send_text(response)
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for user {user_id}")
    except Exception as e:
        print(f"Error handling WebSocket for user {user_id}: {str(e)}")
