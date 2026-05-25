class AppContext:
    """
    A container class for managing global application dependencies and settings.
    """
    def __init__(
        self,
        *,
        settings,
        api_service,
    ):
        self.settings = settings
        self.api_service = api_service
