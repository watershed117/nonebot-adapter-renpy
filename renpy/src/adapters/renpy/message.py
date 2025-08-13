import json
from typing import Type, Iterable
from typing_extensions import override, Self

from nonebot.utils import escape_tag
from nonebot.adapters import Message as BaseMessage
from nonebot.adapters import MessageSegment as BaseMessageSegment


class MessageSegment(BaseMessageSegment["Message"]):
    @classmethod
    @override
    def get_message_class(cls) -> Type["Message"]:
        # 返回适配器的 Message 类型本身
        return Message

    @override
    def __str__(self) -> str:
        """
        返回消息段的纯文本表现形式，适用于日志输出。

        如果是纯文本消息，直接返回转义后的文本内容；
        否则，使用字典推导式安全转换所有值，并返回字符串形式。
        """
        return str(self.data)
        if self.is_text():
            return escape_tag(self.data.get("text", ""))

        # 使用字典推导式 + try-except 安全转换所有值
        safe_data = {
            k: str(v) if not isinstance(
                v, Exception) else f"<{type(v).__name__}>"
            for k, v in self.data.items()
        }
        return escape_tag(str(safe_data))

    @override
    def is_text(self) -> bool:
        # 判断该消息段是否为纯文本
        return self.type == "text"

    @classmethod
    def text(cls, text: str) -> Self:
        return cls("text", {"text": text})

    @classmethod
    def image(cls, format: str, data: str) -> Self:
        return cls("image", {"format": format, "data": data})

    @classmethod
    def audio(cls, format: str, data: str) -> Self:
        return cls("audio", {"format": format, "data": data})

    @classmethod
    def json(cls, data: str) -> Self:
        return cls("json", {"data": data})


class Message(BaseMessage[MessageSegment]):
    @classmethod
    @override
    def get_segment_class(cls) -> Type[MessageSegment]:
        # 返回适配器的 MessageSegment 类型本身
        return MessageSegment

    @staticmethod
    @override
    def _construct(msg: str) -> Iterable[MessageSegment]:
        return [MessageSegment.text(msg)]
