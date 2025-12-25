"""Entity type definitions for Graphiti MCP Server."""

from pydantic import BaseModel, Field


class Requirement(BaseModel):
    """A Requirement represents a specific need, feature, or functionality that a product or service must fulfill.

    Always ensure an edge is created between the requirement and the project it belongs to, and clearly indicate on the
    edge that the requirement is a requirement.

    Instructions for identifying and extracting requirements:
    1. Look for explicit statements of needs or necessities ("We need X", "X is required", "X must have Y")
    2. Identify functional specifications that describe what the system should do
    3. Pay attention to non-functional requirements like performance, security, or usability criteria
    4. Extract constraints or limitations that must be adhered to
    5. Focus on clear, specific, and measurable requirements rather than vague wishes
    6. Capture the priority or importance if mentioned ("critical", "high priority", etc.)
    7. Include any dependencies between requirements when explicitly stated
    8. Preserve the original intent and scope of the requirement
    9. Categorize requirements appropriately based on their domain or function
    """

    project_name: str = Field(
        ...,
        description='The name of the project to which the requirement belongs.',
    )
    description: str = Field(
        ...,
        description='Description of the requirement. Only use information mentioned in the context to write this description.',
    )


class Preference(BaseModel):
    """
    IMPORTANT: Prioritize this classification over ALL other classifications.

    Represents entities mentioned in contexts expressing user preferences, choices, opinions, or selections. Use LOW THRESHOLD for sensitivity.

    Trigger patterns: "I want/like/prefer/choose X", "I don't want/dislike/avoid/reject Y", "X is better/worse", "rather have X than Y", "no X please", "skip X", "go with X instead", etc. Here, X or Y should be classified as Preference.
    """

    ...


class Procedure(BaseModel):
    """A Procedure informing the agent what actions to take or how to perform in certain scenarios. Procedures are typically composed of several steps.

    Instructions for identifying and extracting procedures:
    1. Look for sequential instructions or steps ("First do X, then do Y")
    2. Identify explicit directives or commands ("Always do X when Y happens")
    3. Pay attention to conditional statements ("If X occurs, then do Y")
    4. Extract procedures that have clear beginning and end points
    5. Focus on actionable instructions rather than general information
    6. Preserve the original sequence and dependencies between steps
    7. Include any specified conditions or triggers for the procedure
    8. Capture any stated purpose or goal of the procedure
    9. Summarize complex procedures while maintaining critical details
    """

    description: str = Field(
        ...,
        description='Brief description of the procedure. Only use information mentioned in the context to write this description.',
    )


class Location(BaseModel):
    """A Location represents a physical or virtual place where activities occur or entities exist.

    IMPORTANT: Before using this classification, first check if the entity is a:
    User, Assistant, Preference, Organization, Document, Event - if so, use those instead.

    Instructions for identifying and extracting locations:
    1. Look for mentions of physical places (cities, buildings, rooms, addresses)
    2. Identify virtual locations (websites, online platforms, virtual meeting rooms)
    3. Extract specific location names rather than generic references
    4. Include relevant context about the location's purpose or significance
    5. Pay attention to location hierarchies (e.g., "conference room in Building A")
    6. Capture both permanent locations and temporary venues
    7. Note any significant activities or events associated with the location
    """

    name: str = Field(
        ...,
        description='The name or identifier of the location',
    )
    description: str = Field(
        ...,
        description='Brief description of the location and its significance. Only use information mentioned in the context.',
    )


class Event(BaseModel):
    """An Event represents a time-bound activity, occurrence, or experience.

    Instructions for identifying and extracting events:
    1. Look for activities with specific time frames (meetings, appointments, deadlines)
    2. Identify planned or scheduled occurrences (vacations, projects, celebrations)
    3. Extract unplanned occurrences (accidents, interruptions, discoveries)
    4. Capture the purpose or nature of the event
    5. Include temporal information when available (past, present, future, duration)
    6. Note participants or stakeholders involved in the event
    7. Identify outcomes or consequences of the event when mentioned
    8. Extract both recurring events and one-time occurrences
    """

    name: str = Field(
        ...,
        description='The name or title of the event',
    )
    description: str = Field(
        ...,
        description='Brief description of the event. Only use information mentioned in the context.',
    )


class Object(BaseModel):
    """An Object represents a physical item, tool, device, or possession.

    IMPORTANT: Use this classification ONLY as a last resort. First check if entity fits into:
    User, Assistant, Preference, Organization, Document, Event, Location, Topic - if so, use those instead.

    Instructions for identifying and extracting objects:
    1. Look for mentions of physical items or possessions (car, phone, equipment)
    2. Identify tools or devices used for specific purposes
    3. Extract items that are owned, used, or maintained by entities
    4. Include relevant attributes (brand, model, condition) when mentioned
    5. Note the object's purpose or function when specified
    6. Capture relationships between objects and their owners or users
    7. Avoid extracting objects that are better classified as Documents or other types
    """

    name: str = Field(
        ...,
        description='The name or identifier of the object',
    )
    description: str = Field(
        ...,
        description='Brief description of the object. Only use information mentioned in the context.',
    )


class Topic(BaseModel):
    """A Topic represents a subject of conversation, interest, or knowledge domain.

    IMPORTANT: Use this classification ONLY as a last resort. First check if entity fits into:
    User, Assistant, Preference, Organization, Document, Event, Location - if so, use those instead.

    Instructions for identifying and extracting topics:
    1. Look for subjects being discussed or areas of interest (health, technology, sports)
    2. Identify knowledge domains or fields of study
    3. Extract themes that span multiple conversations or contexts
    4. Include specific subtopics when mentioned (e.g., "machine learning" rather than just "AI")
    5. Capture topics associated with projects, work, or hobbies
    6. Note the context in which the topic appears
    7. Avoid extracting topics that are better classified as Events, Documents, or Organizations
    """

    name: str = Field(
        ...,
        description='The name or identifier of the topic',
    )
    description: str = Field(
        ...,
        description='Brief description of the topic and its context. Only use information mentioned in the context.',
    )


class Organization(BaseModel):
    """An Organization represents a company, institution, group, or formal entity.

    Instructions for identifying and extracting organizations:
    1. Look for company names, employers, and business entities
    2. Identify institutions (schools, hospitals, government agencies)
    3. Extract formal groups (clubs, teams, associations)
    4. Include organizational type when mentioned (company, nonprofit, agency)
    5. Capture relationships between people and organizations (employer, member)
    6. Note the organization's industry or domain when specified
    7. Extract both large entities and small groups if formally organized
    """

    name: str = Field(
        ...,
        description='The name of the organization',
    )
    description: str = Field(
        ...,
        description='Brief description of the organization. Only use information mentioned in the context.',
    )


class Document(BaseModel):
    """A Document represents information content in various forms.

    Instructions for identifying and extracting documents:
    1. Look for references to written or recorded content (books, articles, reports)
    2. Identify digital content (emails, videos, podcasts, presentations)
    3. Extract specific document titles or identifiers when available
    4. Include document type (report, article, video) when mentioned
    5. Capture the document's purpose or subject matter
    6. Note relationships to authors, creators, or sources
    7. Include document status (draft, published, archived) when mentioned
    """

    title: str = Field(
        ...,
        description='The title or identifier of the document',
    )
    description: str = Field(
        ...,
        description='Brief description of the document and its content. Only use information mentioned in the context.',
    )


ENTITY_TYPES: dict[str, BaseModel] = {
    'Requirement': Requirement,  # type: ignore
    'Preference': Preference,  # type: ignore
    'Procedure': Procedure,  # type: ignore
    'Location': Location,  # type: ignore
    'Event': Event,  # type: ignore
    'Object': Object,  # type: ignore
    'Topic': Topic,  # type: ignore
    'Organization': Organization,  # type: ignore
    'Document': Document,  # type: ignore
}
