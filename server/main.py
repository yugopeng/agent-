import json
import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.react_agent import ReactAgent

app = FastAPI()
agent = ReactAgent()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str | None = None
    query: str


async def event_generator(request: Request, req: ChatRequest):
    try:
        for event in agent.execute_stream(req.query, thread_id=req.session_id or "default"):
            if await request.is_disconnected():
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield 'data: {"type":"done"}\n\n'
    except Exception as e:
        yield f'data: {json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)}\n\n'


@app.post("/api/chat/stream")
async def chat_stream(request: Request, req: ChatRequest):
    return StreamingResponse(
        event_generator(request, req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # Nginx 下必须，否则流式会变"憋一大段突然全出"
        },
    )