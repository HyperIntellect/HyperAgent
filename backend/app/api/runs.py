"""API endpoints for run ledger replay/resume/cancel."""

from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import CurrentUser, get_current_user
from app.services.run_ledger import run_ledger_service

router = APIRouter(prefix="/runs")


@router.get("/{run_id}")
async def get_run(run_id: str, current_user: CurrentUser = Depends(get_current_user)):
    run = await run_ledger_service.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.get("user_id") and run["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    return run


@router.get("/{run_id}/timeline")
async def get_run_timeline(run_id: str, current_user: CurrentUser = Depends(get_current_user)):
    run = await run_ledger_service.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.get("user_id") and run["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    timeline = await run_ledger_service.get_timeline(run_id)
    return {"run_id": run_id, "events": timeline}


@router.post("/{run_id}/resume")
async def resume_run(run_id: str, current_user: CurrentUser = Depends(get_current_user)):
    run = await run_ledger_service.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.get("user_id") and run["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    await run_ledger_service.resume_run(run_id)
    return {"success": True, "run_id": run_id, "status": "running"}


@router.post("/{run_id}/cancel")
async def cancel_run(run_id: str, current_user: CurrentUser = Depends(get_current_user)):
    run = await run_ledger_service.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.get("user_id") and run["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    await run_ledger_service.cancel_run(run_id)
    return {"success": True, "run_id": run_id, "status": "cancelled"}

