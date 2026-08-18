from __future__ import annotations

from abc import ABC, abstractmethod

from app.rag.models import RetrievalOutcome



class BaseRetriever(ABC):
    """
    Base RAG retrieval interface.


    Implementations:

        - Qdrant
        - Elasticsearch
        - OpenSearch
        - PostgreSQL pgvector
        - Hybrid Search


    Retrieval sources:


        1. Public Legal Corpus

            legal_documents

            Available:
                all users



        2. Private User Documents

            chatbot_documents

            Security:

                user_id required



        3. User Memory

            user_memory

            Security:

                user_id required



    Isolation model:


        User

          |

          +-- Matters

          |      |

          |      +-- Matter Documents

          |

          +-- Conversations

                 |

                 +-- Conversation Documents



    Every implementation MUST enforce:

        user_id isolation

        matter visibility

        conversation visibility
    """



    @abstractmethod
    async def search(
        self,
        *,
        query: str,
        user_id: str,
        matter_id: str | None = None,
        conversation_id: str | None = None,
        task: str | None = None,
        document_id: str | None = None,
        limit: int = 10,
        filters: dict[str, str] | None = None,
    ) -> RetrievalOutcome:
        """
        Retrieve relevant RAG chunks.


        Parameters:

            query:
                rewritten search query


            user_id:
                tenant security boundary


            matter_id:
                optional matter scope


            conversation_id:
                optional conversation scope


            task:
                legal task classification


            document_id:
                optional uploaded document scope


            limit:
                maximum returned chunks



        Returns:

            RetrievalOutcome
        """

        raise NotImplementedError