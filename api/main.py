"""
API endpoints для ADHDLearningCompanion
"""

from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import json
import logging
from ..utils import ServiceManager, Database

# Инициализация FastAPI
app = FastAPI(
    title="ADHDLearningCompanion API",
    description="API для работы с ADHDLearningCompanion",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене заменить на конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация сервисов
service_manager = ServiceManager()
database = Database()
logger = logging.getLogger(__name__)


# Модели данных
class ProcessRequest(BaseModel):
    """Модель запроса на обработку контента."""

    type: str
    content: Dict[str, Any]
    priority: Optional[str] = "medium"


class UserProfile(BaseModel):
    """Модель профиля пользователя."""

    preferences: Dict[str, Any]
    learning_history: list = []


class ScheduleRequest(BaseModel):
    """Модель запроса на создание расписания."""

    preferences: Dict[str, Any]


# Зависимости
async def get_current_user(authorization: str = Depends(lambda x: x)) -> str:
    """Получение текущего пользователя из токена."""
    try:
        # В реальном приложении здесь будет проверка JWT токена
        return "test_user"  # Временное решение
    except Exception as e:
        raise HTTPException(
            status_code=401, detail="Invalid authentication credentials"
        )


# Endpoints
@app.post("/api/process")
async def process_content(
    request: ProcessRequest, user_id: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """Обработка контента."""
    try:
        result = await service_manager.handle_request(user_id, request.dict())
        return result
    except Exception as e:
        logger.error(f"Ошибка при обработке контента: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/profile/{user_id}")
async def get_profile(
    user_id: str, current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """Получение профиля пользователя."""
    if user_id != current_user:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        profile = await database.get_user_profile(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        return profile
    except Exception as e:
        logger.error(f"Ошибка при получении профиля: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/profile/{user_id}")
async def update_profile(
    user_id: str, profile: UserProfile, current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """Обновление профиля пользователя."""
    if user_id != current_user:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        updated_profile = await database.update_user_profile(user_id, profile.dict())
        return updated_profile
    except Exception as e:
        logger.error(f"Ошибка при обновлении профиля: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/schedule/{user_id}")
async def create_schedule(
    user_id: str,
    request: ScheduleRequest,
    current_user: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """Создание персонализированного расписания."""
    if user_id != current_user:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        # Обновляем предпочтения перед созданием расписания
        await database.update_user_profile(
            user_id, {"preferences": request.preferences}
        )
        schedule = await service_manager.scheduler.create_personalized_schedule(user_id)
        return schedule
    except Exception as e:
        logger.error(f"Ошибка при создании расписания: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/{user_id}")
async def get_analytics(
    user_id: str,
    event_type: Optional[str] = None,
    current_user: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """Получение аналитики пользователя."""
    if user_id != current_user:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        analytics = await database.get_user_analytics(user_id, event_type)
        return {"analytics": analytics}
    except Exception as e:
        logger.error(f"Ошибка при получении аналитики: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket для real-time обновлений
class ConnectionManager:
    """Менеджер WebSocket подключений."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)


manager = ConnectionManager()


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint для real-time обновлений."""
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Обработка входящих сообщений
            await manager.send_personal_message(
                json.dumps({"status": "received", "data": data}), user_id
            )
    except WebSocketDisconnect:
        manager.disconnect(user_id)


# Запуск приложения
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
