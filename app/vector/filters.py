from __future__ import annotations

from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MinShould,
)


class QdrantFilterBuilder:
    """
    Central Qdrant security filter builder.


    Security hierarchy:

    User
     |
     +-- User Memory
     |
     +-- Matters
     |      |
     |      +-- Conversations
     |
     +-- Documents


    HARD SECURITY RULE:

        Every private vector MUST contain:

            user_id


    Visibility:


    scope=user
    -----------
        Available everywhere
        for same user.


    scope=matter
    -------------
        Available only inside
        same matter.


    scope=conversation
    --------------------
        Available only inside
        same conversation.

    """



    # ==================================================
    # User Isolation Filter
    # ==================================================

    @staticmethod
    def user_scope(
        *,
        user_id: str,
    ) -> Filter:
        """
        Base tenant isolation.

        Use for every private collection query.
        """


        return Filter(

            must=[

                FieldCondition(

                    key="user_id",

                    match=MatchValue(

                        value=user_id

                    ),

                )

            ]

        )



    # ==================================================
    # Document Visibility
    # ==================================================

    @staticmethod
    def document_visibility(
        *,
        user_id: str,
        matter_id: str | None = None,
        conversation_id: str | None = None,
    ) -> Filter:
        """
        Visibility rules for:

            chatbot_documents


        Allows:

            user documents

            OR

            same matter documents

            OR

            same conversation documents


        Always restricted by user_id.
        """

        visibility: list[Filter] = []


        # ----------------------------------
        # User level documents
        # ----------------------------------

        visibility.append(

            Filter(

                must=[

                    FieldCondition(

                        key="scope",

                        match=MatchValue(

                            value="user"

                        ),

                    )

                ]

            )

        )



        # ----------------------------------
        # Matter documents
        # ----------------------------------

        if matter_id:

            visibility.append(

                Filter(

                    must=[

                        FieldCondition(

                            key="scope",

                            match=MatchValue(

                                value="matter"

                            ),

                        ),

                        FieldCondition(

                            key="matter_id",

                            match=MatchValue(

                                value=matter_id

                            ),

                        ),

                    ]

                )

            )



        # ----------------------------------
        # Conversation documents
        # ----------------------------------

        if conversation_id:

            visibility.append(

                Filter(

                    must=[

                        FieldCondition(

                            key="scope",

                            match=MatchValue(

                                value="conversation"

                            ),

                        ),

                        FieldCondition(

                            key="conversation_id",

                            match=MatchValue(

                                value=conversation_id

                            ),

                        ),

                    ]

                )

            )



        return Filter(

            must=[

                FieldCondition(

                    key="user_id",

                    match=MatchValue(

                        value=user_id

                    ),

                )

            ],


            min_should=MinShould(

                conditions=visibility,

                min_count=1,

            ),

        )

    # ==================================================
    # Single uploaded document (strict isolation)
    # ==================================================

    @staticmethod
    def document_id_scoped(
        *,
        user_id: str,
        document_id: str,
        matter_id: str | None = None,
        conversation_id: str | None = None,
    ) -> Filter:
        """
        Restrict retrieval to one uploaded document.

        Always requires:
            user_id + document_id

        Also applies the same visibility scopes used for
        chatbot_documents so matter/conversation isolation
        is preserved.
        """
        must = [
            FieldCondition(
                key="user_id",
                match=MatchValue(value=user_id),
            ),
            FieldCondition(
                key="document_id",
                match=MatchValue(value=document_id),
            ),
        ]

        visibility: list[Filter] = [
            Filter(
                must=[
                    FieldCondition(
                        key="scope",
                        match=MatchValue(value="user"),
                    )
                ]
            )
        ]

        if matter_id:
            visibility.append(
                Filter(
                    must=[
                        FieldCondition(
                            key="scope",
                            match=MatchValue(value="matter"),
                        ),
                        FieldCondition(
                            key="matter_id",
                            match=MatchValue(value=matter_id),
                        ),
                    ]
                )
            )

        if conversation_id:
            visibility.append(
                Filter(
                    must=[
                        FieldCondition(
                            key="scope",
                            match=MatchValue(value="conversation"),
                        ),
                        FieldCondition(
                            key="conversation_id",
                            match=MatchValue(
                                value=conversation_id
                            ),
                        ),
                    ]
                )
            )

        return Filter(
            must=must,
            min_should=MinShould(
                conditions=visibility,
                min_count=1,
            ),
        )

    # ==================================================
    # Conversation-only documents
    # ==================================================

    @staticmethod
    def conversation_documents(
        *,
        user_id: str,
        conversation_id: str,
    ) -> Filter:
        return Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id),
                ),
                FieldCondition(
                    key="scope",
                    match=MatchValue(value="conversation"),
                ),
                FieldCondition(
                    key="conversation_id",
                    match=MatchValue(value=conversation_id),
                ),
            ]
        )

    # ==================================================
    # Matter-only documents
    # ==================================================

    @staticmethod
    def matter_documents(
        *,
        user_id: str,
        matter_id: str,
    ) -> Filter:
        return Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id),
                ),
                FieldCondition(
                    key="scope",
                    match=MatchValue(value="matter"),
                ),
                FieldCondition(
                    key="matter_id",
                    match=MatchValue(value=matter_id),
                ),
            ]
        )

    # ==================================================
    # Legal corpus metadata filters
    # ==================================================

    @staticmethod
    def legal_metadata(
        filters: dict[str, str] | None = None,
    ) -> Filter | None:
        if not filters:
            return None

        must: list[FieldCondition] = []

        year = filters.get("year")
        if year:
            try:
                must.append(
                    FieldCondition(
                        key="year",
                        match=MatchValue(value=int(year)),
                    )
                )
            except ValueError:
                must.append(
                    FieldCondition(
                        key="year",
                        match=MatchValue(value=year),
                    )
                )

        court = filters.get("court")
        if court:
            must.append(
                FieldCondition(
                    key="court",
                    match=MatchValue(value=court),
                )
            )

        section = filters.get("section")
        if section:
            must.append(
                FieldCondition(
                    key="sections",
                    match=MatchValue(value=section),
                )
            )

        if not must:
            return None

        return Filter(must=must)



    # ==================================================
    # Memory Visibility
    # ==================================================

    @staticmethod
    def memory_visibility(
        *,
        user_id: str,
        matter_id: str | None = None,
        conversation_id: str | None = None,
    ) -> Filter:
        """
        Visibility rules for:

            user_memory


        Memory hierarchy:

            User memory

            Matter memory

            Conversation memory


        Always restricted by user_id.
        """

        visibility: list[Filter] = []



        # ----------------------------------
        # Global user memory
        # ----------------------------------

        visibility.append(

            Filter(

                must=[

                    FieldCondition(

                        key="scope",

                        match=MatchValue(

                            value="user"

                        ),

                    )

                ]

            )

        )



        # ----------------------------------
        # Matter memory
        # ----------------------------------

        if matter_id:

            visibility.append(

                Filter(

                    must=[

                        FieldCondition(

                            key="scope",

                            match=MatchValue(

                                value="matter"

                            ),

                        ),

                        FieldCondition(

                            key="matter_id",

                            match=MatchValue(

                                value=matter_id

                            ),

                        ),

                    ]

                )

            )



        # ----------------------------------
        # Conversation memory
        # ----------------------------------

        if conversation_id:

            visibility.append(

                Filter(

                    must=[

                        FieldCondition(

                            key="scope",

                            match=MatchValue(

                                value="conversation"

                            ),

                        ),

                        FieldCondition(

                            key="conversation_id",

                            match=MatchValue(

                                value=conversation_id

                            ),

                        ),

                    ]

                )

            )



        return Filter(

            must=[

                FieldCondition(

                    key="user_id",

                    match=MatchValue(

                        value=user_id

                    ),

                )

            ],

            min_should=MinShould(

                conditions=visibility,

                min_count=1,

            ),

        )