import asyncio

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from graph.builder import build_graph
from memory.vector_store import init_vectorstore
from settings.config import settings


async def run_agent():
    init_vectorstore()
    graph = build_graph()

    await graph.ainvoke({})


async def run_pipeline():
    init_vectorstore()

    hour, minute = settings.schedule_time.split(":")

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        run_agent,
        trigger=CronTrigger(hour=int(hour), minute=int(minute)),
    )
    scheduler.start()
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(run_pipeline())
