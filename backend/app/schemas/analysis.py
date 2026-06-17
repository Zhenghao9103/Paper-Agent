from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PaperAnalysisRead(BaseModel):
    id: int
    document_id: int
    summary_zh: str
    innovations: str
    methodology: str
    experiments: str
    limitations: str
    chart_insights: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
