from typing_extensions import override

from nonebot.compat import model_dump
from nonebot.adapters import Event as BaseEvent
from nonebot.utils import escape_tag

class Event(BaseEvent):
    event_type : str
    event_name : str
    data : dict

    @override
    def get_event_name(self) -> str:
        # 返回事件的名称，用于日志打印
        return self.event_name

    @override
    def get_event_description(self) -> str:
        # 返回事件的描述，用于日志打印，请注意转义 loguru tag
        return escape_tag(repr(model_dump(self)))

    @override
    def get_message(self):
        # 获取事件消息的方法，根据事件具体实现，如果事件非消息类型事件，则抛出异常
        raise ValueError("Event has no message!")

    @override
    def get_user_id(self) -> str:
        # 获取用户 ID 的方法，根据事件具体实现，如果事件没有用户 ID，则抛出异常
        raise ValueError("Event has no context!")

    @override
    def get_session_id(self) -> str:
        # 获取事件会话 ID 的方法，根据事件具体实现，如果事件没有相关 ID，则抛出异常
        raise ValueError("Event has no context!")

    @override
    def is_tome(self) -> bool:
        # 判断事件是否和机器人有关
        return False
    
