from fastapi import APIRouter, status

from graph_service.dto import MetricsResponse
from graph_service.routers.ingest import async_worker

router = APIRouter()


@router.get('/metrics', status_code=status.HTTP_200_OK)
async def get_metrics():
    """
    Get current system metrics including queue size and worker status.
    """
    return MetricsResponse(
        queue_size=async_worker.get_queue_size(),
        worker_status=async_worker.get_status()
    )
