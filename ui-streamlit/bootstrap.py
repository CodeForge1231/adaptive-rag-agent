from core.config import Settings
from core.context import AppContext
from services.api_service import ApiService


def bootstrap():
    """
    Initialize and wire together the core application components.

    Returns
    -------
    AppContext
        A fully initialized application context containing settings 
        and service instances.
    """
    # Load application configuration from environment variables
    settings = Settings()

    # Initialize the primary API communication service
    api_service = ApiService(
        base_url=settings.base_api_url,
        api_token=settings.api_token,
    )

    # Assemble and return the dependency container
    return AppContext(
        settings=settings,
        api_service=api_service,
    )
