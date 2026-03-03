"""Router for file library management."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import CurrentUser, get_current_user
from app.core.logging import get_logger
from app.db.base import get_db
from app.db.models import File as FileModel
from app.services.file_storage import file_storage_service

logger = get_logger(__name__)

router = APIRouter(prefix="/library")


@router.get("")
async def list_library_files(
    content_type: Optional[str] = Query(
        None, description="Filter by content type prefix, e.g. image/*"
    ),
    search: Optional[str] = Query(None, description="Search by filename substring"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user),
):
    """List all files for the authenticated user."""

    # Base query filtered by user
    base_filter = [FileModel.user_id == current_user.id]

    # Content type filter (supports wildcard like "image/*")
    if content_type:
        if content_type.endswith("/*"):
            prefix = content_type[:-1]  # "image/" from "image/*"
            base_filter.append(FileModel.content_type.startswith(prefix))
        else:
            base_filter.append(FileModel.content_type == content_type)

    # Search filter (filename substring, case-insensitive)
    if search:
        base_filter.append(FileModel.original_filename.ilike(f"%{search}%"))

    # Count total matching files
    count_query = select(func.count()).select_from(FileModel).where(*base_filter)
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Fetch files with pagination
    query = (
        select(FileModel)
        .where(*base_filter)
        .order_by(FileModel.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(query)
    files = result.scalars().all()

    # Build response with presigned URLs
    file_list = []
    for f in files:
        preview_url = await file_storage_service.get_presigned_url(f.storage_key)
        file_list.append(
            {
                "id": f.id,
                "filename": f.original_filename,
                "content_type": f.content_type,
                "file_size": f.file_size,
                "storage_key": f.storage_key,
                "created_at": f.created_at.isoformat() if f.created_at else None,
                "preview_url": preview_url,
            }
        )

    return {"files": file_list, "total": total}


@router.delete("/{file_id}")
async def delete_library_file(
    file_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Delete a file from the library."""

    result = await db.execute(
        select(FileModel).where(
            FileModel.id == file_id,
            FileModel.user_id == current_user.id,
        )
    )
    file_record = result.scalar_one_or_none()

    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")

    await file_storage_service.delete_file(file_record.storage_key)
    await db.delete(file_record)
    await db.commit()

    return {"status": "deleted", "file_id": file_id}
