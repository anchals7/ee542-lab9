
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from passlib.context import CryptContext
from sqlalchemy.exc import IntegrityError


PROJECT_DIR = Path(__file__).resolve().parents[2]
HEARTWISE_DIR = PROJECT_DIR / "embedding" / "heartwise"
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

load_dotenv(PROJECT_DIR / ".env")

sys.path.insert(0, str(HEARTWISE_DIR))

from chat_heartwise import (
    check_emergency,
    generate_answer,
    load_resources,
    retrieve,
)

DATABASE_URL = os.getenv("DATABASE_URL")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"

if not DATABASE_URL or not JWT_SECRET:
    raise RuntimeError("DATABASE_URL and JWT_SECRET must be configured.")

engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)

JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "43200"))

password_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)


class AuthRequest(BaseModel):
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1)


def make_token(user_id: int, username: str) -> str:
    expires = datetime.now(timezone.utc) + timedelta(
        minutes=JWT_EXPIRE_MINUTES
    )

    return jwt.encode(
        {
            "sub": str(user_id),
            "username": username,
            "exp": expires,
        },
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    publisher: str
    title: str
    url: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    emergency: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the FAISS index and embedding model only once.
    app.state.index, app.state.metadata, app.state.model = (
        await asyncio.to_thread(load_resources)
    )

    yield

    await engine.dispose()


app = FastAPI(title="HeartWise API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8002", "http://127.0.0.1:8002"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


async def get_current_user(authorization: str | None = Header(default=None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Login required.")

    scheme, separator, token = authorization.partition(" ")

    if not separator or scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Invalid authorization header.")

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = int(payload["sub"])
    except (JWTError, ValueError, KeyError, TypeError):
        raise HTTPException(status_code=401, detail="Invalid or expired token.")

    async with engine.connect() as connection:
        result = await connection.execute(
            text("SELECT id, username FROM users WHERE id = :user_id"),
            {"user_id": user_id},
        )
        user = result.mappings().first()

    if not user:
        raise HTTPException(status_code=401, detail="User not found.")

    return dict(user)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "HeartWise"}

@app.post("/api/signup")
async def signup(request: AuthRequest):
    username = request.username.strip()

    if not username:
        raise HTTPException(
            status_code=422,
            detail="Username cannot be empty.",
        )

    password_hash = password_context.hash(request.password)

    try:
        async with engine.begin() as connection:
            result = await connection.execute(
                text(
                    """
                    INSERT INTO users (username, password_hash)
                    VALUES (:username, :password_hash)
                    """
                ),
                {
                    "username": username,
                    "password_hash": password_hash,
                },
            )

            user_id = result.lastrowid

    except IntegrityError:
        raise HTTPException(
            status_code=409,
            detail="Username already exists.",
        )

    return {
        "token": make_token(user_id, username),
        "username": username,
    }


@app.post("/api/login")
async def login(request: AuthRequest):
    async with engine.connect() as connection:
        result = await connection.execute(
            text(
                """
                SELECT id, username, password_hash
                FROM users
                WHERE username = :username
                """
            ),
            {"username": request.username.strip()},
        )

        user = result.mappings().first()

    if not user or not password_context.verify(
        request.password,
        user["password_hash"],
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
        )

    return {
        "token": make_token(user["id"], user["username"]),
        "username": user["username"],
    }

@app.post("/api/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    user=Depends(get_current_user),
):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    emergency_answer = check_emergency(question)

    if emergency_answer:
        return AskResponse(
            answer=emergency_answer,
            sources=[],
            emergency=True,
        )

    try:
        retrieved = await asyncio.to_thread(
            retrieve,
            question,
            app.state.index,
            app.state.metadata,
            app.state.model,
        )

        answer = await asyncio.to_thread(
            generate_answer,
            question,
            retrieved,
        )

    except httpx.HTTPError:
        raise HTTPException(
            status_code=503,
            detail="The local LLM server is unavailable.",
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="HeartWise could not process the question.",
        )

    sources = []
    seen_urls = set()

    for chunk in retrieved:
        url = chunk["source_url"]

        if url not in seen_urls:
            sources.append(
                Source(
                    publisher=chunk["publisher"],
                    title=chunk["title"],
                    url=url,
                )
            )
            seen_urls.add(url)

    return AskResponse(
        answer=answer,
        sources=sources,
        emergency=False,
    )


@app.get("/")
async def home():
    index_file = FRONTEND_DIR / "index.html"

    if not index_file.exists():
        return {"message": "HeartWise API is running. Frontend coming soon."}

    return FileResponse(index_file)


app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static",
)

