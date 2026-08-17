from graph_service.wikis.templates import PROJECT_WIKI_ENTITY_TYPES


def test_optional_collection_attributes_accept_absent_source_values() -> None:
    nullable_fields = {
        'Project': 'owner',
        'Product': 'owner',
        'ProductModule': 'owner',
        'ProductFeature': 'owner',
        'Requirement': 'owner',
        'Defect': 'owner',
        'Person': 'role',
    }

    for entity_name, field_name in nullable_fields.items():
        model = PROJECT_WIKI_ENTITY_TYPES[entity_name]
        assert getattr(model.model_validate({field_name: None}), field_name) is None


def test_optional_collection_attributes_preserve_explicit_values() -> None:
    for entity_name, field_name in {'Project': 'owner', 'Person': 'role'}.items():
        model = PROJECT_WIKI_ENTITY_TYPES[entity_name]
        assert getattr(model.model_validate({field_name: ['张三']}), field_name) == ['张三']
