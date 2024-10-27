import GPUtil
from fastapi import APIRouter, logger
from pydantic import BaseModel
router = APIRouter(
    prefix="/gpu",
    tags=["model"],
    dependencies=[],
    responses={404: {"description": "Not found"}},
)

class GPU_Info(BaseModel):
    id: int
    name: str
    load: float
    memory_free: float
    memory_used: float
    memory_total: float
    temperature: float

class Response(BaseModel):
    code: int
    message: str
    data: GPU_Info



@router.get("/gpu_usage")
def get_gpu_info():
    try:
        gpus = GPUtil.getGPUs()
        gpu = gpus[0]
        return Response(
            code=200,
            message="Success",
            data=GPU_Info(
                id=gpu.id,
                name=gpu.name,
                load=gpu.load,
                memory_free=gpu.memoryFree,
                memory_used=gpu.memoryUsed,
                memory_total=gpu.memoryTotal,
                temperature=gpu.temperature
            )
        )
    except Exception as e:
        logger.error(f"Error while getting GPU info: {e}")
        return Response(
            code=500,
            message=f"Error while getting GPU info: {e}",
            data=None
        )

    # for gpu in gpus:
    #     print(f"GPU ID: {gpu.id}")
    #     print(f"GPU Name: {gpu.name}")
    #     print(f"GPU Load: {gpu.load * 100}%")
    #     print(f"GPU Free Memory: {gpu.memoryFree}MB")
    #     print(f"GPU Used Memory: {gpu.memoryUsed}MB")
    #     print(f"GPU Total Memory: {gpu.memoryTotal}MB")
    #     print(f"GPU Temperature: {gpu.temperature}°C")
    #     print("-" * 30)