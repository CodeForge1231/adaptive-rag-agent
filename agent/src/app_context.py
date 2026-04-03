import asyncio


class AppContext:
    def __init__(
        self,
        *,
        settings: dict,
    ):
        self.settings = settings

    async def shutdown(self):
        if self.persistence is not None:
            await self.persistence.shutdown()
        loop = asyncio.get_running_loop()
        await loop.shutdown_default_executor()
