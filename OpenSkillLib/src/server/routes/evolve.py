from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

class EvolveRequest(BaseModel):
    skill_id: str
    tasks: list[str]
    analyst_model: Optional[str] = None

@router.post("/{skill_id}")
async def evolve_skill(skill_id: str, req: EvolveRequest, request: Request):
    client = request.app.state.client
    try:
        result = await client.evolve(
            skill_id=skill_id,
            tasks=req.tasks,
            analyst_model=req.analyst_model
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))