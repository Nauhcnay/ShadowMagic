import asyncio
from concurrent.futures import ProcessPoolExecutor
import functools

import test_controlnet

executor_batch = ProcessPoolExecutor(1)
executor_interactive = ProcessPoolExecutor(1)

async def run_async( executor, f ):
    ## We expect this to be called from inside an existing loop.
    ## As a result, we call `get_running_loop()` instead of `get_event_loop()` so that
    ## it raises an error if our assumption is false, rather than creating a new loop.
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor( executor, f )
    return data

async def run_single( *args, **kwargs ):
    return await run_async( executor_batch, functools.partial( test_controlnet.run_single, *args, **kwargs ) )