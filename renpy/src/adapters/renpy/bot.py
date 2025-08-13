import base64
import pathlib
from typing import Any, Union
from typing_extensions import override

from nonebot.message import handle_event
from nonebot.adapters import Bot as BaseBot

from .event import Event
from .message import Message, MessageSegment

from nonebot.adapters import Adapter
from nonebot import logger


class Bot(BaseBot):
    @override
    def __init__(self, adapter: Adapter, self_id: str, **kwargs: Any):
        super().__init__(adapter, self_id)
        self.adapter: Adapter = adapter
        # 一些有关 Bot 的信息也可以在此定义和存储

    async def handle_event(self, event: Event):
        # 根据需要，对事件进行某些预处理，例如：
        # 检查事件是否和机器人有关操作，去除事件消息首尾的 @bot
        # 检查事件是否有回复消息，调用平台 API 获取原始消息的消息内容
        ...
        # 调用 handle_event 让 NoneBot 对事件进行处理
        await handle_event(self, event)

    @override
    async def send(
        self,
        event: Event,
        message: Union[str, Message, MessageSegment],  # 接受多种类型
        **kwargs: Any,
    ) -> Any:
        """
        发送消息到 Ren'Py 客户端。
        构造一个 JSON 对象，将其序列化为字符串，然后发送。
        """
        if isinstance(message, Message):
            parts = []
            for segment in message:
                if segment.is_text():
                    parts.append({"text": segment.data.get("text", "")})
                else:
                    parts.append({"type": segment.type, "data": segment.data})

        elif isinstance(message, MessageSegment):
            if message.is_text():
                parts = [{"text": message.data.get("text", "")}]
            else:
                parts = [{"type": message.type, "data": message.data}]
        else:
            parts = [{"text": str(message)}]

        final_payload = {}
        for part in parts:
            final_payload.update(part)
            final_payload.update(kwargs)
        try:
            logger.info(f"[{self.self_id}] Sending message: {final_payload}")
            return await self.adapter._call_api(self, "send", **final_payload)
        except Exception as e:
            logger.error(
                f"[{self.self_id}] Failed to send message over WebSocket: {e}")

    async def show(self,image:bytes|pathlib.Path,name:str):
        if isinstance(image,bytes):
            image_bytes = base64.b64encode(image).decode("utf-8")
            return await self.adapter._call_api(self,"show",bytestring=image_bytes,name=name)
        if isinstance(image,pathlib.Path):
            return await self.adapter._call_api(self,"show",path=image.as_posix(),name=name)
        else:
            raise TypeError("image must be bytes or pathlib.Path")

