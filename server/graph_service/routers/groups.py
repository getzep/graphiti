from fastapi import APIRouter

from graph_service.dto import ResolveGroupIdRequest, ResolveGroupIdResponse
from graph_service.group_ids import resolve_group_id

router = APIRouter()


@router.post('/groups/resolve')
async def resolve_group(request: ResolveGroupIdRequest) -> ResolveGroupIdResponse:
    return ResolveGroupIdResponse(group_id=resolve_group_id(request.scope, request.key))
