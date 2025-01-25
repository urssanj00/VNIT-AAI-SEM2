import time
from fastapi import FastAPI
import asyncio
import datetime
app = FastAPI()





@app.get("/greeting")
async def greeting(data):
    nw = datetime.datetime.now()
    start_time = nw.strftime("%d-%m-%y %H:%M:%S")

    print(f'{data}->start_time {start_time}')
    await asyncio.sleep(10)
    nw = datetime.datetime.now()
    end_time = nw.strftime("%d-%m-%y %H:%M:%S")  #  total_time = end_time - start_time  # Calculate epoch time
    print(f'{data}->end_time: {end_time}')
    return {"message": f"{data}Hello World FastAPI"}
