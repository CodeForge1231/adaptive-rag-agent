from .base import BaseFilterAdapter


class ChromaFilterAdapter(BaseFilterAdapter):
    """
    Adapter for converting generic filter dictionaries into ChromaDB syntax.
    """

    def transform(self, filters: dict):
        """
        Translate generic filters into a structured ChromaDB filter object.

        Parameters
        ----------
        filters : dict
            The raw filter criteria from the application layer.

        Returns
        -------
        dict or None
            A dictionary formatted for Chroma's `where` parameter, 
            or None if no valid filters exist.
        """
        if not filters:
            return None

        # Remove empty / None values
        clean_filters = {
            k: v for k, v in filters.items()
            if v is not None
        }

        if not clean_filters:
            return None

        conditions = []

        for key, value in clean_filters.items():

            # Filter by topic list
            if key == "topic":
                if isinstance(value, list):
                    conditions.append({"topic": {"$in": value}})
                else:
                    conditions.append({"topic": value})

            # Exclude specific document IDs
            elif key == "exclude_ids" and isinstance(value, list) and value:
                conditions.append({
                    "id": {"$nin": [str(v) for v in value]}
                })

            # Direct match fields
            elif key in {"age_group", "level"}:
                conditions.append({key: value})

        # No valid conditions produced
        if not conditions:
            return None

        # Single condition
        if len(conditions) == 1:
            return conditions[0]

        # Multiple conditions
        return {"$and": conditions}