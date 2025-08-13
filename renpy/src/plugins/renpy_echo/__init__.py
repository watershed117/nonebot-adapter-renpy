from typing import List
from nonebot import on_message
from nonebot import logger
import random
import pathlib
import base64

from ...adapters.renpy.event import MessageEvent
from ...adapters.renpy.bot import Bot

echo = on_message(priority=1, block=False)


def escape_renpy_tag(text: str | dict):
    if isinstance(text, dict):
        return str(text).replace("{", "{{")
    return text.replace("{", "\\{")


def random_image(path: pathlib.Path, image_format: str = "jpg", recursion: bool = False) -> pathlib.Path:
    if not path.exists():
        raise FileNotFoundError(f"The specified path does not exist: {path}")
    if not path.is_dir():
        raise FileNotFoundError(
            f"The specified path is not a directory: {path}")

    suffix = image_format.lower().lstrip('.')
    target_suffix = f".{suffix}"
    image_files: List[pathlib.Path] = []
    if recursion:
        image_files = [f for f in path.rglob(
            f'*{target_suffix}') if f.is_file()]
    else:
        for item in path.iterdir():
            if item.is_file() and item.suffix.lower() == target_suffix:
                image_files.append(item)
    if not image_files:
        search_scope = "directory and its subdirectories" if recursion else "directory"
        raise ValueError(
            f"No images with format '{image_format}' found in the specified {search_scope}: {path}")
    return random.choice(image_files)


def byte_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


@echo.handle()
async def handle_echo(event: MessageEvent, bot: Bot):
    message = event.get_data()
    logger.info(f"Received message: {message}")
    text = str(message.get("text"))
    if text:
        if text == "image":
            image = random_image(pathlib.Path(
                r"C:\Users\water\Desktop\renpy\Ushio_Noa\game\images\background"), recursion=True)
            logger.info(f"Sending image: {image}")
            with open(image, "rb") as f:
                image_bytes = f.read()
            await bot.show(image=image_bytes,name=image.name)
            # await bot.show(image=image.__str__(), name=image.name)
        else:
            await bot.say(who="e", what=escape_renpy_tag(text))
    else:
        await echo.finish()
