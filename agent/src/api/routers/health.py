from fastapi import APIRouter

router = APIRouter(prefix="/api")


@router.get("/health")
def health():
    """
    Perform a basic health check of the application.

    Returns
    -------
    dict
        A simple dictionary indicating the operational status:
        - 'status' (str): Always returns 'ok' if the service is up.
    """
    return {"status": "ok"}
