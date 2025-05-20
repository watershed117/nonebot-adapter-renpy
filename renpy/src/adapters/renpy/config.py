from pydantic import Field, BaseModel


class Config(BaseModel):
    bot_id : str
    ws_url : str
    ws_name : str