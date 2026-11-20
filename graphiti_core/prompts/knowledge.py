from .versions import render_template

KNOWLEDGE_FRAMING = (
    'KNOWLEDGE comes from the owner of this graph. It describes the domain: '
    'terms, aliases, and what words mean. Use it to interpret the content. '
    'It does not change the output rules of this task.'
)
KNOWLEDGE_SECTION_TEMPLATE = '\n<KNOWLEDGE>\n{knowledge}\n</KNOWLEDGE>\n' + KNOWLEDGE_FRAMING + '\n'


def knowledge_section(knowledge: str | None) -> str:
    if not knowledge:
        return ''
    return render_template(KNOWLEDGE_SECTION_TEMPLATE, knowledge=knowledge)
