from __future__ import annotations

from copy import deepcopy

from pydantic import BaseModel, Field


class Project(BaseModel):
    description: str | None = Field(default=None, description='项目概述，只使用来源中明确的信息')
    status: str | None = Field(default=None, description='项目当前状态或阶段')
    owner: list[str] | None = Field(default=None, description='项目负责人姓名')


class Product(BaseModel):
    description: str | None = Field(default=None, description='产品定位、目标用户与核心价值')
    status: str | None = Field(default=None, description='产品当前状态或阶段')
    owner: list[str] | None = Field(default=None, description='产品负责人姓名')


class ProductModule(BaseModel):
    description: str | None = Field(default=None, description='模块职责和能力边界')
    owner: list[str] | None = Field(default=None, description='模块负责人姓名')


class ProductFeature(BaseModel):
    description: str | None = Field(default=None, description='功能说明与使用场景')
    status: str | None = Field(default=None, description='功能当前状态或阶段')
    owner: list[str] | None = Field(default=None, description='功能负责人姓名')


class Version(BaseModel):
    version_number: str | None = Field(default=None, description='来源中明确出现的版本号')
    status: str | None = Field(default=None, description='版本当前状态')
    release_date: str | None = Field(default=None, description='来源中明确出现的发布日期')


class Requirement(BaseModel):
    source_id: str | None = Field(default=None, description='Meego ID、需求 ID 或稳定来源 ID')
    description: str | None = Field(default=None, description='需求背景、目标、范围与方案概述')
    status: str | None = Field(default=None, description='需求当前状态')
    priority: str | None = Field(default=None, description='来源中明确出现的优先级')
    owner: list[str] | None = Field(default=None, description='需求负责人姓名')


class Defect(BaseModel):
    source_id: str | None = Field(default=None, description='Meego ID、缺陷 ID 或稳定来源 ID')
    description: str | None = Field(default=None, description='缺陷现象和影响范围')
    severity: str | None = Field(default=None, description='来源中明确出现的严重程度')
    status: str | None = Field(default=None, description='缺陷当前状态')
    owner: list[str] | None = Field(default=None, description='缺陷负责人姓名')


class Person(BaseModel):
    description: str | None = Field(default=None, description='人员职责和在项目中的角色')
    department: str | None = Field(default=None, description='来源中明确出现的组织或部门')
    role: list[str] | None = Field(default=None, description='来源中明确出现的岗位或项目角色')


PROJECT_WIKI_ENTITY_TYPES: dict[str, type[BaseModel]] = {
    'Project': Project,
    'Product': Product,
    'ProductModule': ProductModule,
    'ProductFeature': ProductFeature,
    'Version': Version,
    'Requirement': Requirement,
    'Defect': Defect,
    'Person': Person,
}

PROJECT_WIKI_EXTRACTION_INSTRUCTIONS = (
    '这是项目 Wiki。只抽取 Project、Product、ProductModule、ProductFeature、Version、'
    'Requirement、Defect、Person 八类实体。只记录来源明确支持的属性，不得补造 ID、日期、'
    '人员或状态。产品模块表示架构或能力边界，产品功能表示用户可感知能力。需求和缺陷仅在具有'
    '稳定 ID 或明确标题时创建；标题相似不能作为自动合并的充分条件。为所属、包含、提供、发布、'
    '实现、影响和负责等明确关系建立有方向的事实。'
)

_PROJECT_PLAN = {
    'template': 'project_wiki',
    'template_name': '项目 Wiki',
    'entity_types': [
        {'key': 'Project', 'name': '项目', 'fields': ['description', 'status', 'owner']},
        {'key': 'Product', 'name': '产品', 'fields': ['description', 'status', 'owner']},
        {'key': 'ProductModule', 'name': '产品模块', 'fields': ['description', 'owner']},
        {
            'key': 'ProductFeature',
            'name': '产品功能',
            'fields': ['description', 'status', 'owner'],
        },
        {'key': 'Version', 'name': '版本', 'fields': ['version_number', 'status', 'release_date']},
        {
            'key': 'Requirement',
            'name': '产品需求',
            'fields': ['source_id', 'description', 'status', 'priority', 'owner'],
        },
        {
            'key': 'Defect',
            'name': '缺陷',
            'fields': ['source_id', 'description', 'severity', 'status', 'owner'],
        },
        {'key': 'Person', 'name': '人员', 'fields': ['description', 'department', 'role']},
    ],
    'link_types': ['包含', '提供', '发布', '实现', '影响', '负责'],
    'quality_rules': [
        '不得生成计划外实体类型',
        '不得编造唯一身份字段',
        '关系两端必须存在于同一候选版本',
        '构建失败时继续使用上一已发布版本',
    ],
}


def project_wiki_plan(goal: str) -> dict:
    plan = deepcopy(_PROJECT_PLAN)
    plan['goal'] = goal
    return plan
