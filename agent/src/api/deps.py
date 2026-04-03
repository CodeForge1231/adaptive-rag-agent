from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.app_context import AppContext

security = HTTPBearer()


def get_context(request: Request) -> AppContext:
    """
    Retrieve the application context from the global app state.

    Parameters
    ----------
    request : Request
        The incoming FastAPI request object.

    Returns
    -------
    AppContext
        The shared application context containing services and settings.
    """
    return request.app.state.context


def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    ctx: AppContext = Depends(get_context),
):
    """
    Validate the bearer token provided in the authorization header.

    Parameters
    ----------
    credentials : HTTPAuthorizationCredentials
        The credentials extracted from the Authorization header.
    ctx : AppContext
        The application context used to retrieve the expected token.

    Returns
    -------
    str
        The validated API token.

    Raises
    ------
    HTTPException
        If the provided token does not match the configured API token.
    """
    token = credentials.credentials

    expected = ctx.settings["app"].get("api_token")

    if expected and token != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API token")

    return token
