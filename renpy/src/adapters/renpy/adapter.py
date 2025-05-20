import json
from typing import Any, Dict
from typing_extensions import override

from pydantic import BaseModel
from yarl import URL

from nonebot import get_plugin_config
from nonebot.utils import logger_wrapper
from nonebot.compat import type_validate_python
from nonebot.drivers import (
    Driver,
    Request,
    Response,
    ASGIMixin,
    WebSocket,
    HTTPServerSetup,
    WebSocketServerSetup,
)
from nonebot.adapters import Adapter as BaseAdapter
from nonebot.exception import WebSocketClosed
from nonebot.utils import escape_tag

from .bot import Bot
from .event import Event
from .config import Config

import asyncio


log = logger_wrapper("Renpy_Adapter")

class Adapter(BaseAdapter):
    @override
    def __init__(self, driver: Driver, **kwargs: Any):
        super().__init__(driver, **kwargs)
        self.adapter_config: Config = get_plugin_config(Config)
        self.setup()
        self.connections :dict[str, WebSocket] = {}  # 记录所有连接的 WebSocket

    @classmethod
    @override
    def get_name(cls) -> str:
        """适配器名称"""
        return "Renpy_Adapter"

    def setup(self) -> None:
        if not isinstance(self.driver, ASGIMixin):
            raise RuntimeError(
                f"Current driver {self.config.driver} doesn't support asgi server!"
                f"{self.get_name()} Adapter need a asgi server driver to work."
            )

        # 反向 Websocket 路由
        ws_setup = WebSocketServerSetup(
            URL(self.adapter_config.ws_url),  # 路由地址
            self.adapter_config.ws_name,  # 路由名称
            self._handle_ws,  # 处理函数
        )
        self.setup_websocket_server(ws_setup)

    async def _handle_ws(self, websocket: WebSocket) -> Any:
        """WebSocket 路由处理函数，只有一个类型为 WebSocket 的参数"""
        await websocket.accept()
        bot = Bot(self, self_id=self.adapter_config.bot_id)  # 实例化 Bot
        self.bot_connect(bot)  # 建立 Bot 连接
        self.connections[bot.self_id] = websocket  # 记录连接
        try:
            while True:
                data = await websocket.receive()
                if data is None:
                    break
                # 处理数据
                data = json.loads(data)
                event = self.payload_to_event(data)
                asyncio.create_task(bot.handle_event(event))
        except WebSocketClosed:
            log("WARNING", f"WebSocket for Bot {escape_tag(bot.self_id)} closed by peer")
        except Exception as e:
                log(
                    "ERROR",
                    "<r><bg #f8bbd0>Error while process data from websocket "
                    f"for bot {escape_tag(bot.self_id)}.</bg #f8bbd0></r>",
                    e,
                )
        finally:
            self.bot_disconnect(bot)  # 断开 Bot 连接
            await websocket.close()
            
    @classmethod
    def payload_to_event(cls, payload: Dict[str, Any]) -> Event:
        """根据平台事件的特性，转换平台 payload 为具体 Event

        Event 模型继承自 pydantic.BaseModel，具体请参考 pydantic 文档
        """

        # 做一层异常处理，以应对平台事件数据的变更
        try:
            return type_validate_python(Event, payload)
        except Exception as e:
            # 无法正常解析为具体 Event 时，给出日志提示
            log(
                "WARNING",
                f"Parse event error: {str(payload)}",
            )
            # 也可以尝试转为基础 Event 进行处理
            return type_validate_python(Event, payload)

    @override
    async def _call_api(self, bot: Bot, api: str, **data: Any) -> Any:
        log("DEBUG", f"Calling API <y>{api}</y>")  # 给予日志提示
        platform_data = json.dumps(data)  # 自行将数据转为平台所需要的格式
        ws: WebSocket = self.connections[bot.self_id]
        await ws.send_text(platform_data)
        return await ws.receive_text()
