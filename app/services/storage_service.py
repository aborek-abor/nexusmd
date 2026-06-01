"""
NexusMD — Storage Service
Handles upload of docking results to a Railway S3-compatible bucket
and generation of signed download URLs.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("nexusmd.storage")

# Support both Railway-native bucket env vars (BUCKET, ENDPOINT, ACCESS_KEY_ID,
# SECRET_ACCESS_KEY) and the conventional AWS_ prefixed names as fallbacks.
BUCKET_NAME = os.environ.get("BUCKET", os.environ.get("BUCKET_NAME", "nexusmd-results"))
BUCKET_REGION = os.environ.get("BUCKET_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.environ.get("ACCESS_KEY_ID", os.environ.get("AWS_ACCESS_KEY_ID", ""))
AWS_SECRET_ACCESS_KEY = os.environ.get("SECRET_ACCESS_KEY", os.environ.get("AWS_SECRET_ACCESS_KEY", ""))
AWS_ENDPOINT_URL = os.environ.get("ENDPOINT", os.environ.get("AWS_ENDPOINT_URL", ""))

# Signed URL expiry in seconds (24 hours)
SIGNED_URL_EXPIRY = 86400


def _get_s3_client():
    """Return a boto3 S3 client configured for the Railway bucket, or None if unavailable."""
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        logger.debug("S3 credentials not configured — bucket upload disabled")
        return None
    try:
        import boto3
        kwargs = dict(
            region_name=BUCKET_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
        if AWS_ENDPOINT_URL:
            kwargs["endpoint_url"] = AWS_ENDPOINT_URL
        return boto3.client("s3", **kwargs)
    except ImportError:
        logger.warning("boto3 not installed — bucket upload disabled")
        return None
    except Exception as e:
        logger.error(f"Failed to create S3 client: {e}")
        return None


async def upload_job_results(job_id: str, job_dir: Path) -> bool:
    """
    Upload all docking result files for a job to the configured S3 bucket.

    Uploads:
      - poses.sdf          → {job_id}/poses.sdf
      - out_*.pdbqt        → {job_id}/out_*.pdbqt
      - lig_*.pdbqt        → {job_id}/lig_*.pdbqt

    Returns True if upload succeeded, False otherwise.
    """
    s3 = _get_s3_client()
    if s3 is None:
        logger.info(f"[Storage] Skipping bucket upload for job {job_id} — S3 not configured")
        return False

    if not job_dir.exists():
        logger.warning(f"[Storage] Job directory not found: {job_dir}")
        return False

    # Collect files to upload
    files_to_upload: list[Path] = []

    poses_sdf = job_dir / "poses.sdf"
    if poses_sdf.exists():
        files_to_upload.append(poses_sdf)

    files_to_upload.extend(sorted(job_dir.glob("out_*.pdbqt")))
    files_to_upload.extend(sorted(job_dir.glob("lig_*.pdbqt")))

    if not files_to_upload:
        logger.warning(f"[Storage] No result files found in {job_dir}")
        return False

    uploaded = 0
    for file_path in files_to_upload:
        s3_key = f"{job_id}/{file_path.name}"
        try:
            s3.upload_file(
                str(file_path),
                BUCKET_NAME,
                s3_key,
                ExtraArgs={"ContentType": "application/octet-stream"},
            )
            logger.info(f"[Storage] Uploaded {file_path.name} → s3://{BUCKET_NAME}/{s3_key}")
            uploaded += 1
        except Exception as e:
            logger.error(f"[Storage] Failed to upload {file_path.name}: {e}")

    logger.info(f"[Storage] Job {job_id}: {uploaded}/{len(files_to_upload)} files uploaded")
    return uploaded > 0


def get_download_url(job_id: str, filename: str) -> Optional[str]:
    """
    Generate a pre-signed S3 download URL for a specific file in a job's bucket prefix.

    Returns the signed URL string, or None if S3 is not configured or the call fails.
    """
    s3 = _get_s3_client()
    if s3 is None:
        return None

    s3_key = f"{job_id}/{filename}"
    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": BUCKET_NAME, "Key": s3_key},
            ExpiresIn=SIGNED_URL_EXPIRY,
        )
        return url
    except Exception as e:
        logger.error(f"[Storage] Failed to generate signed URL for {s3_key}: {e}")
        return None
