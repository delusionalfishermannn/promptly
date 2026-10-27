"""Publishing: packs, listings, storefront publish and rollback."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sqlalchemy import select

from backend.deps import RequestContext, require
from backend.schemas import PublishRequest, RollbackRequest
from core.artifacts import get_store
from core.errors import NotFound
from core.rbac import Permission
from db.models import Artifact, PromptProduct, PromptVersion
from governance.gate import publish_version, rollback_version
from publishing.listings import build_listing, suggest_price
from publishing.packager import build_prompt_pack
from publishing.storefront import list_adapters

router = APIRouter(tags=["publishing"])


@router.get("/storefronts")
def storefronts(
    context: Annotated[RequestContext, Depends(require(Permission.READ_CATALOG))],
) -> dict[str, Any]:
    return {"items": list_adapters()}


@router.get("/products/{product_id}/listing")
def listing(
    product_id: str,
    context: Annotated[RequestContext, Depends(require(Permission.READ_CATALOG))],
) -> dict[str, Any]:
    product = context.session.get(PromptProduct, product_id)
    if product is None:
        raise NotFound(f"product {product_id} not found")
    version = (
        context.session.get(PromptVersion, product.current_version_id)
        if product.current_version_id
        else None
    )
    if version is None:
        version = context.session.scalar(
            select(PromptVersion)
            .where(PromptVersion.product_id == product.id)
            .order_by(PromptVersion.created_at.desc())
            .limit(1)
        )
    if version is None:
        raise NotFound("product has no versions")
    generated = build_listing(
        context.session,
        product=product,
        version=version,
        price_aud=product.price_aud_micros / 1_000_000,
    )
    return {
        "product_id": product.id,
        "version_id": version.id,
        "stored_listing": product.listing,
        "generated_listing": generated,
        "pricing_suggestion": suggest_price(
            segment=product.buyer,
            hypothesis_aud=product.price_aud_micros / 1_000_000,
            confidence=0.5,
        ),
    }


@router.post("/versions/{version_id}/pack")
def build_pack(
    version_id: str,
    context: Annotated[RequestContext, Depends(require(Permission.MANAGE_PRODUCTS))],
) -> dict[str, Any]:
    version = context.session.get(PromptVersion, version_id)
    if version is None:
        raise NotFound(f"version {version_id} not found")
    product = context.session.get(PromptProduct, version.product_id)
    if product is None:
        raise NotFound("product not found")
    pack = build_prompt_pack(
        context.session, version=version, product=product, actor_id=context.actor_id
    )
    context.audit(
        "packaging.built",
        subject_type="version",
        subject_id=version.id,
        decision="built",
        detail={"sha256": pack.sha256, "signature": pack.signature, "files": len(pack.files)},
    )
    return pack.to_dict()


@router.post("/versions/{version_id}/publish")
def publish(
    version_id: str,
    payload: PublishRequest,
    context: Annotated[RequestContext, Depends(require(Permission.PUBLISH))],
) -> dict[str, Any]:
    return publish_version(
        context.session,
        version_id=version_id,
        actor_id=context.actor_id,
        price_aud=payload.price_aud,
    )


@router.post("/products/{product_id}/rollback")
def rollback(
    product_id: str,
    payload: RollbackRequest,
    context: Annotated[RequestContext, Depends(require(Permission.PUBLISH))],
) -> dict[str, Any]:
    return rollback_version(
        context.session,
        product_id=product_id,
        to_version_id=payload.to_version_id,
        actor_id=context.actor_id,
        reason=payload.reason,
    )


@router.get("/versions/{version_id}/pack")
def pack_info(
    version_id: str,
    context: Annotated[RequestContext, Depends(require(Permission.READ_CATALOG))],
) -> dict[str, Any]:
    version = context.session.get(PromptVersion, version_id)
    if version is None:
        raise NotFound(f"version {version_id} not found")
    artifacts = list(
        context.session.scalars(
            select(Artifact)
            .where(Artifact.subject_id == version.id, Artifact.kind == "prompt_pack")
            .order_by(Artifact.created_at.desc())
        )
    )
    return {
        "version_id": version.id,
        "signature": version.signature,
        "signed_at": version.signed_at.isoformat() if version.signed_at else None,
        "artifact_uri": version.artifact_uri,
        "artifacts": [
            {
                "uri": row.uri,
                "sha256": row.sha256,
                "size_bytes": row.size_bytes,
                "created_at": row.created_at.isoformat(),
                "signature": row.signature,
            }
            for row in artifacts
        ],
        "store": type(get_store()).__name__,
    }
