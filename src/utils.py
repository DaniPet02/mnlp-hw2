# Import Datases to work with Transformers by Hugging-Face
from pydantic import BaseModel

class EvaluationGride(BaseModel):
    short_comment: str
    proposed_traslation: str
    score: int