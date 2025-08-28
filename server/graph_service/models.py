from pydantic import BaseModel, Field


class Team(BaseModel):
    team: str | None = Field(..., description="The name of the team")


class Project(BaseModel):
    """A project"""

    project: str | None = Field(..., description="General project name")


class KubeResource(BaseModel):
    """A Kubernetes resource"""

    namespace: str | None = Field(..., description="The namespace of the resource")
    kind: str | None = Field(..., description="The kind of the resource")
    component: str | None = Field(..., description="The component of the resource")


class ContainerImage(BaseModel):
    """A container image"""

    image: str | None = Field(..., description="The name of the image")
    pipelineLink: str | None = Field(
        ..., description="The TeamCity link of the pipeline"
    )
    pipelineConfigId: str | None = Field(
        ..., description="The TeamCity build config id of the pipeline"
    )


class Cluster(BaseModel):
    """A Kubernetes cluster"""

    cluster: str | None = Field(..., description="The name of the cluster")


class GitRepo(BaseModel):
    """A Git repository"""

    org: str | None = Field(..., description="The organization of the repository")
    url: str | None = Field(..., description="The URL of the repository")


class IsMemberOf(BaseModel):
    """A relationship between a parent and a child"""


class IsGitOpsRepoFile(BaseModel):
    """A relationship between a parent and a child"""

    file_path: str | None = Field(
        ..., description="The file path of the resource in the GitRepo"
    )
    github_link: str | None = Field(..., description="The GitHub link of the resource")


class IsOwnedBy(BaseModel):
    """A resource is owned by a team"""


class IsCodeRepo(BaseModel):
    """A relationship between a parent and a child"""


class IsDeployedTo(BaseModel):
    """A relationship between a parent and a child"""


class IsInstanceOf(BaseModel):
    """A relationship between a parent and a child"""

    instance: str | None = Field(..., description="The name of the instance")
    component: str | None = Field(..., description="The component of the instance")
    portal_link: str | None = Field(..., description="The HTTP link of the portal page")


entity_types = {
    "Project": Project,
    "Team": Team,
    "KubeResource": KubeResource,
    "ContainerImage": ContainerImage,
    "Cluster": Cluster,
    "GitRepo": GitRepo,
}
edge_types = {
    "IsMemberOf": IsMemberOf,
    "IsOwnedBy": IsOwnedBy,
    "IsGitOpsRepoFile": IsGitOpsRepoFile,
    "IsCodeRepo": IsCodeRepo,
    "IsDeployedTo": IsDeployedTo,
    "IsInstanceOf": IsInstanceOf,
}
edge_type_map = {
    ("Project", "Team"): ["IsOwnedBy"],
    ("KubeResource", "Project"): ["IsInstanceOf"],
    ("KubeResource", "Cluster"): ["IsDeployedTo"],
    ("Project", "GitRepo"): ["IsGitOpsRepoFile", "IsCodeRepo"],
    ("Project", "Cluster"): ["IsDeployedTo"],
    ("ContainerImage", "KubeResource"): ["IsDeployedTo"],
    ("Entity", "Entity"): ["RELATES_TO"],
}
