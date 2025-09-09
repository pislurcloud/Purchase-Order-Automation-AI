def transform_to_schema(obj):
    schema_obj = {
        'order_id': obj.get('order_id'),
        'client_name': obj.get('client_name'),
        'order_date': obj.get('order_date'),
        'delivery_date': obj.get('delivery_date'),
        'items': obj.get('items', []),
        'order_total': obj.get('order_total'),
        'currency': obj.get('currency'),
        'special_instructions': obj.get('special_instructions'),
        'confidence_score': obj.get('confidence_score', 0.0),
        'validation_errors': []
    }
    try:
        items_total = sum([float(i.get('total_price',0) or 0) for i in schema_obj['items']])
        if schema_obj['order_total'] not in (None, items_total):
            schema_obj['validation_errors'].append('order_total_mismatch')
    except Exception:
        pass
    return schema_obj
