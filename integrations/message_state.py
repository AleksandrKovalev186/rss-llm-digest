from operator import add
from typing import Annotated

from langchain_core.messages import AnyMessage
from pydantic import Field, BaseModel


class State(BaseModel):
    messages: Annotated[list[AnyMessage], add] = Field(default_factory=list)

    summaries: str = Field(
        default="",
        description="summary from agents"
    )
