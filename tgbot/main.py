import asyncio
import logging

from aiogram import Bot, Dispatcher
from roters import mainHandler

from tokens import TOKEN

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

bot = Bot(TOKEN)
dp = Dispatcher()
dp.include_router(mainHandler.router)

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(dp.start_polling(bot))
