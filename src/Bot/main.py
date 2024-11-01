from fastapi import FastAPI, WebSocket


# FastAPI app initialization
app = FastAPI()

# Instantiate the chatbot
chatbot = Chatbot()


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()
    while True:
        # Receive the user's message
        data = await websocket.receive_text()

        # Get response from the chatbot
        response = chatbot.generate_response(user_id, data)

        # Send the response back to the client
        await websocket.send_text(response)
