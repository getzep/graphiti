from pydantic import BaseModel, Field, field_validator


class WikiCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    slug: str | None = Field(default=None, pattern=r'^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$')
    goal: str = Field(default='', max_length=2000)
    data_scope: str = Field(default='specified', pattern=r'^(all|specified)$')

    @field_validator('name')
    @classmethod
    def strip_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('name 不能为空')
        return value

    @field_validator('goal')
    @classmethod
    def strip_goal(cls, value: str) -> str:
        return value.strip()


class WikiSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    max_facts: int = Field(default=10, ge=1, le=50)
