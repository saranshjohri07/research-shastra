from dataclasses import dataclass

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver


@dataclass
class CheckpointerResources:
    pool: ConnectionPool
    saver: PostgresSaver


def build_checkpoint_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("backend.app.retrieval.retriever", "RetrievedEvidence"),
        ]
    )


def build_checkpointer(database_url: str) -> CheckpointerResources:
    pool = ConnectionPool(
        conninfo=database_url,
        min_size=1,
        max_size=5,
        open=True,
        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
        },
    )
    try:
        saver = PostgresSaver(conn=pool, serde=build_checkpoint_serializer())
        saver.setup()
    except BaseException:
        pool.close()
        raise
    return CheckpointerResources(pool=pool, saver=saver)


def close_checkpointer(resources: CheckpointerResources) -> None:
    resources.pool.close()
