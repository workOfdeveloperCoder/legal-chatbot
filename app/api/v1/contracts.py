from uuid import UUID

from fastapi import APIRouter, Body, Depends, Query
from fastapi.responses import Response

from app.api.dependencies.auth import get_current_active_user
from app.api.dependencies.services import get_contract_analysis_service
from app.contracts.fields import DEFAULT_CLAUSE_FIELDS
from app.models.user import User
from app.schemas.contracts import (
    ClauseExtractionResponse,
    ClauseFieldSpec,
    ClauseFieldsResponse,
    ExportRequest,
    PlaybookCatalogItem,
    PlaybookReviewRequest,
    PlaybookReviewResponse,
    RedlineRequest,
    RedlineResponse,
    ReviewTableRequest,
    ReviewTableResponse,
)
from app.services.contract_analysis_service import ContractAnalysisService

router = APIRouter(
    prefix="/contracts",
    tags=["Contracts"],
)


@router.get(
    "/fields",
    response_model=ClauseFieldsResponse,
    response_model_by_alias=True,
)
async def list_clause_fields(
    current_user: User = Depends(get_current_active_user),
):
    del current_user
    return ClauseFieldsResponse(
        fields=[
            ClauseFieldSpec(
                key=field.key,
                label=field.label,
                question=field.question,
            )
            for field in DEFAULT_CLAUSE_FIELDS
        ]
    )


@router.get(
    "/playbooks",
    response_model=list[PlaybookCatalogItem],
    response_model_by_alias=True,
)
async def list_playbooks(
    current_user: User = Depends(get_current_active_user),
    service: ContractAnalysisService = Depends(get_contract_analysis_service),
):
    del current_user
    return service.catalog_playbooks()


@router.post(
    "/documents/{document_id}/clauses",
    response_model=ClauseExtractionResponse,
    response_model_by_alias=True,
)
async def extract_document_clauses(
    document_id: UUID,
    current_user: User = Depends(get_current_active_user),
    service: ContractAnalysisService = Depends(get_contract_analysis_service),
):
    return await service.extract_document(
        user_id=current_user.id,
        document_id=document_id,
    )


@router.post(
    "/documents/{document_id}/playbook-review",
    response_model=PlaybookReviewResponse,
    response_model_by_alias=True,
)
async def playbook_review_document(
    document_id: UUID,
    payload: PlaybookReviewRequest = Body(default=PlaybookReviewRequest()),
    current_user: User = Depends(get_current_active_user),
    service: ContractAnalysisService = Depends(get_contract_analysis_service),
):
    return await service.playbook_review_document(
        user_id=current_user.id,
        document_id=document_id,
        playbook_id=payload.playbook_id,
        rules=payload.rules,
    )


@router.post(
    "/documents/{document_id}/redline",
    response_model=RedlineResponse,
    response_model_by_alias=True,
)
async def redline_document(
    document_id: UUID,
    payload: RedlineRequest = Body(default=RedlineRequest()),
    current_user: User = Depends(get_current_active_user),
    service: ContractAnalysisService = Depends(get_contract_analysis_service),
):
    if payload.against_document_id is not None:
        return await service.redline_documents(
            user_id=current_user.id,
            document_id=document_id,
            against_document_id=payload.against_document_id,
        )
    return await service.redline_playbook(
        user_id=current_user.id,
        document_id=document_id,
        playbook_id=payload.playbook_id,
        rules=payload.rules,
    )


@router.post(
    "/export",
)
async def export_docx(
    payload: ExportRequest,
    current_user: User = Depends(get_current_active_user),
    service: ContractAnalysisService = Depends(get_contract_analysis_service),
):
    content, filename = await service.export_bytes(
        user_id=current_user.id,
        kind=payload.kind,
        document_id=payload.document_id,
        against_document_id=payload.against_document_id,
        matter_id=payload.matter_id,
        conversation_id=payload.conversation_id,
        playbook_id=payload.playbook_id,
        title=payload.title,
        body=payload.body,
    )
    return Response(
        content=content,
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@router.post(
    "/matters/{matter_id}/review-table",
    response_model=ReviewTableResponse,
    response_model_by_alias=True,
)
async def build_matter_review_table(
    matter_id: UUID,
    payload: ReviewTableRequest = Body(default=ReviewTableRequest()),
    current_user: User = Depends(get_current_active_user),
    service: ContractAnalysisService = Depends(get_contract_analysis_service),
):
    return await service.review_table(
        user_id=current_user.id,
        matter_id=matter_id,
        field_keys=payload.field_keys,
        extra_questions=payload.extra_questions,
        document_ids=payload.document_ids,
    )


@router.post(
    "/conversations/{conversation_id}/review-table",
    response_model=ReviewTableResponse,
    response_model_by_alias=True,
)
async def build_conversation_review_table(
    conversation_id: UUID,
    payload: ReviewTableRequest = Body(default=ReviewTableRequest()),
    current_user: User = Depends(get_current_active_user),
    service: ContractAnalysisService = Depends(get_contract_analysis_service),
    matter_id: UUID | None = Query(default=None),
):
    return await service.review_table(
        user_id=current_user.id,
        conversation_id=conversation_id,
        matter_id=matter_id,
        field_keys=payload.field_keys,
        extra_questions=payload.extra_questions,
        document_ids=payload.document_ids,
    )
