from jsonschema import validate, ValidationError

schema = {
    'type': 'object',
    'properties': {
        'order_id': {'type': ['string','null']},
        'client_name': {'type': ['string','null']},
        'order_date': {'type': ['string','null']},
        'delivery_date': {'type': ['string','null']},
        'items': {'type': 'array'},
        'order_total': {'type': ['number','null']},
        'currency': {'type': ['string','null']},
        'confidence_score': {'type': 'number'}
    },
    'required': ['items','confidence_score']
}

def validate_obj(obj):
    try:
        validate(instance=obj, schema=schema)
        return True, []
    except ValidationError as e:
        return False, [str(e)]
