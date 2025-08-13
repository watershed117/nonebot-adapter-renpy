import nonebot
from src.adapters.renpy.adapter import Adapter

nonebot.init()

driver = nonebot.get_driver()

driver.register_adapter(adapter=Adapter)
nonebot.load_builtin_plugins('echo')


nonebot.load_from_toml("pyproject.toml")

if __name__ == "__main__":
    nonebot.run()