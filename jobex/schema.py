"""Схема структурованої вакансії (Pydantic v2).

Вимоги кейсу: >= 6 полів, >= 1 enum, >= 1 вкладений об'єкт/список.
Тут: 12 полів верхнього рівня, 8 enum-ів, вкладені об'єкти Salary і списки Skill / LanguageRequirement.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Seniority(str, Enum):
    intern = "intern"
    junior = "junior"
    middle = "middle"
    senior = "senior"
    lead = "lead"
    head = "head"
    unknown = "unknown"


class RoleCategory(str, Enum):
    backend = "backend"
    frontend = "frontend"
    fullstack = "fullstack"
    mobile = "mobile"
    qa = "qa"
    devops = "devops"
    data_ml = "data_ml"
    erp_1c = "erp_1c"
    design = "design"
    management = "management"
    marketing = "marketing"
    sales = "sales"
    recruitment_hr = "recruitment_hr"
    support = "support"
    other = "other"


class WorkFormat(str, Enum):
    remote = "remote"
    office = "office"
    hybrid = "hybrid"
    flexible = "flexible"  # кандидат обирає між кількома форматами
    unknown = "unknown"


class EmploymentType(str, Enum):
    full_time = "full_time"
    part_time = "part_time"
    contract = "contract"
    freelance = "freelance"
    internship = "internship"
    unknown = "unknown"


class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    UAH = "UAH"
    PLN = "PLN"
    other = "other"


class SalaryPeriod(str, Enum):
    hour = "hour"
    month = "month"
    year = "year"
    project = "project"


class LanguageLevel(str, Enum):
    A1 = "A1"
    A2 = "A2"
    B1 = "B1"
    B2 = "B2"
    C1 = "C1"
    C2 = "C2"
    native = "native"
    unknown = "unknown"


class PostingLanguage(str, Enum):
    uk = "uk"
    en = "en"
    ru = "ru"
    mixed = "mixed"


class Salary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: Optional[float] = Field(..., ge=0, description="Нижня межа; null, якщо вказана лише верхня ('до $700').")
    max: Optional[float] = Field(..., ge=0, description="Верхня межа; null, якщо вказана лише нижня ('від 500$').")
    currency: Currency
    period: SalaryPeriod = Field(description="Якщо період не вказано — month.")

    @model_validator(mode="after")
    def _check_range(self) -> "Salary":
        if self.min is None and self.max is None:
            raise ValueError("salary must have at least one of min/max; use salary=null instead")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"salary.min ({self.min}) > salary.max ({self.max})")
        return self


class Skill(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Технологія/інструмент/хард-скіл у канонічному написанні, напр. 'PostgreSQL'.")
    required: bool = Field(description="true — обов'язкова вимога; false — 'буде плюсом' / nice to have.")


class LanguageRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str = Field(pattern=r"^[A-Za-z]{2}$", description="ISO 639-1 код (дві латинські літери): en, uk, ru, pl, de ...")
    level: LanguageLevel

    @field_validator("language")
    @classmethod
    def _lower(cls, v: str) -> str:
        return v.lower()


class JobPosting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=2, description="Назва позиції без зарплати/локації.")
    company: Optional[str] = Field(..., description="Назва компанії; null, якщо не вказана.")
    role_category: RoleCategory
    seniority: Seniority
    work_format: WorkFormat
    employment_type: EmploymentType
    locations: list[str] = Field(..., description="Міста/країни/регіони англійською: 'Kyiv', 'Ukraine', 'EU', 'Worldwide'.")
    salary: Optional[Salary] = Field(..., description="null, якщо чисел немає ('$$$$', 'обговорюється').")
    experience_years_min: Optional[float] = Field(..., ge=0, le=40, description="Мінімум років досвіду; 0 — 'без досвіду'; null — не вказано.")
    skills: list[Skill] = Field(..., description="[] якщо навичок немає.")
    languages: list[LanguageRequirement] = Field(..., description="[] якщо мовних вимог немає.")
    posting_language: PostingLanguage


def json_schema() -> dict:
    """JSON Schema, яку передаємо моделям (tool input_schema / Ollama format)."""
    return JobPosting.model_json_schema()


FIELDS: list[str] = list(JobPosting.model_fields.keys())


def inline_refs(schema: dict) -> dict:
    """Розгортає $ref/$defs у плоску схему — деякі бекенди (граматики Ollama) так працюють надійніше."""
    import copy

    defs = schema.get("$defs", {})

    def _resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                resolved = _resolve(copy.deepcopy(defs[name]))
                extra = {k: v for k, v in node.items() if k != "$ref"}
                return {**resolved, **extra}
            return {k: _resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [_resolve(x) for x in node]
        return node

    return _resolve(schema)
