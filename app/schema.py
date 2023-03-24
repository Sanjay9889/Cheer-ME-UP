from pydantic import BaseModel
from pydantic.typing import Literal

# Shared properties
class CheerMeUpBase(BaseModel):
    age: float
    gender: Literal["Male", "Female", "Others"]
    one_word_question: dict
    subjective_question: dict


class CheerMeUpCreate(CheerMeUpBase):
    pass


class CheerMeUpCreateResponse(CheerMeUpBase):
    class Config:
        orm_mode = True
