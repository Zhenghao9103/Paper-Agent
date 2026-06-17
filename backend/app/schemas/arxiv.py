from pydantic import BaseModel


class ArxivPaper(BaseModel):
    title: str
    authors: list[str]
    summary: str
    published: str
    pdf_url: str | None
    entry_url: str
