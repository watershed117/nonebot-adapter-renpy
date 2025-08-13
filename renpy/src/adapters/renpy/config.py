from nonebot import get_driver
from pydantic import BaseModel, Field

class Config(BaseModel):
    """Renpy 适配器配置"""
    bot_id: str = Field("renpy_bot", description="机器人的唯一ID")
    ws_url: str = Field("ws://localhost:20000/ws", description="WebSocket连接URL")
    ws_name: str = Field("renpy", description="WebSocket路由名称")
    api_timeout: int = Field(10, description="API请求超时时间（秒）")
