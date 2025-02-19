import time
from fastapi import FastAPI
import asyncio
import datetime

app = FastAPI()


@app.get("/greeting")
async def greeting(name: str):
    nw = datetime.datetime.now()
    start_time = nw.strftime("%d-%m-%y %H:%M:%S")

    print(f"{name} -> Start Time: {start_time}")

    await asyncio.sleep(10)  # Simulating a delay

    nw = datetime.datetime.now()
    end_time = nw.strftime("%d-%m-%y %H:%M:%S")

    print(f"{name} -> End Time: {end_time}")

    return {
        "message": f"Hello {name}, welcome to FastAPI!",
        "start_time": start_time,
        "end_time": end_time
    }
