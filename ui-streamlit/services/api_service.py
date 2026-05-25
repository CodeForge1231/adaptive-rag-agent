import httpx


class ApiService:
    """
    A service class to interact with a remote API for chat and document management.
    """

    def __init__(self, base_url: str, api_token: str):
        self.api_token = api_token
        self.base_url = base_url

    def _headers(self):
        """
        Construct the authorization headers for the request.

        Returns
        -------
        dict
            A dictionary containing the 'Authorization' bearer token.
        """
        return {"Authorization": f"Bearer {self.api_token}"}

    async def ask_question(
        self,
        *,
        payload,
    ):
        """
        Send a question query to the chat endpoint.

        Parameters
        ----------
        payload : dict
            The data to be sent in the JSON body of the POST request.
            Typically includes the question text and context parameters.

        Returns
        -------
        httpx.Response
            The response object from the API.

        Notes
        -----
        This method uses an asynchronous HTTP client with a 60-second timeout.
        """
        # Initialize asynchronous client with base configuration
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=60,
        ) as client:
            return await client.post(
                "/chat/ask",
                json=payload,
                headers=self._headers(),
            )

    async def add_document(self, *, payload):
        """
        Upload or register a document via the API.

        Parameters
        ----------
        payload : dict
            The document metadata and content to be sent as JSON.

        Returns
        -------
        httpx.Response
            The response object from the API.

        Raises
        ------
        httpx.HTTPError
            If the request fails due to network or server issues.
        """
        # Use context manager to ensure the client session is closed properly
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=60,
        ) as client:
            # Send document data to the processing endpoint
            return await client.post(
                "/documents/add",
                json=payload,
                headers=self._headers(),
            )
