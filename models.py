from typing import List, Dict
from pydantic import BaseModel

class ThoughtStage(BaseModel):
    """Structure for individual thought stages"""
    name: str
    content: str

class ThoughtProcess(BaseModel):
    """Complete thought process with multiple stages"""
    stages: List[ThoughtStage]
    
    def to_dict(self) -> Dict[str, str]:
        return {stage.name: stage.content for stage in self.stages}
    
class ModelInit(BaseModel):
    model_number: int = 1
    temperature: float = 0.7
    streaming: bool = True