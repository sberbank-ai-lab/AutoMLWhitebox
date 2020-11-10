from typing import Dict, List
from io import StringIO


def preprocess_features(model_data: Dict, table_name: str) -> List[str]:
    """

    Args:
        model_data:
        table_name:

    Returns:

    """
    result = ['SELECT']

    str_type = 'VARCHAR(50)'

    for i, feature in enumerate(model_data['features']):
        woe = model_data['features'][feature]

        if woe['f_type'] == 'real':
            string = f"(CASE WHEN {feature} IS NULL OR {feature} = 'NaN' THEN NULL ELSE {feature} END) AS {feature}"
        else:
            big_cats, spec_value = woe['spec_cat']
            big_cats_str = ','.join(map(lambda x: f"'{x}'", big_cats))
            string = f"""(CASE WHEN {feature} IS NULL OR LOWER(CAST({feature} AS {str_type})) = 'nan' 
THEN NULL WHEN {feature} NOT IN ({big_cats_str}) THEN '{spec_value}' ELSE CAST({feature} AS {str_type}) END) AS {feature}"""

        is_not_final = i != len(model_data['features']) - 1
        result.append('    ' + string + (',' if is_not_final else ''))

    result.append(f'FROM {table_name}')
    return result


def transform_features(model_data: Dict, from_source: List[str]) -> List[str]:
    """

    Args:
        model_data:
        from_source:

    Returns:

    """
    result = ['SELECT']

    for i, feature in enumerate(model_data['features']):
        woe = model_data['features'][feature]
        nan_value = woe['nan_value']

        if woe['f_type'] == 'real':
            if len(woe['splits']) == 0:
                string = f"{woe['cod_dict'][0]}"
            else:
                splits = [(woe['splits'][i], woe['cod_dict'][i]) for i in range(len(woe['splits']))]
                map_str = ' '.join([f'WHEN {feature} <= {x[0]} THEN {x[1]}' for x in splits])
                else_case = woe['cod_dict'][len(woe['splits'])]
                nan_case = woe['spec_cod'][nan_value] if nan_value in woe['spec_cod'] else 0.0
                string = f"CASE WHEN {feature} IS NULL THEN {nan_case} {map_str} ELSE {else_case} END"
        else:
            if len(woe['cat_map']) == 0:
                string = f"{woe['cod_dict'][0]}"
            else:
                cat_map = [(cat, woe['cod_dict'][woe_idx]) for cat, woe_idx in woe['cat_map'].items()]
                map_str = ' '.join([f"WHEN {feature} = '{x[0]}' THEN {x[1]}" for x in cat_map])
                spec_case = ' '.join(map(lambda x: f"WHEN {feature} = '{x[0]}' THEN {x[1]}", woe['spec_cod'].items()))
                if nan_value in woe['spec_cod']:
                    nan_case = woe['spec_cod'][nan_value]
                elif nan_value in woe['cat_map']:
                    nan_case = woe['cod_dict'][woe['cat_map'][nan_value]]
                else:
                    nan_case = 0.0
                string = f"CASE WHEN {feature} IS NULL THEN {nan_case} {spec_case} {map_str} ELSE {nan_case} END"

        is_not_final = i != len(model_data['features']) - 1
        result.append(f"    ({string}) AS {feature}" + (',' if is_not_final else ''))

    result.append(f'FROM (')

    for s in from_source:
        result.append(f'    {s}')

    result.append(f')')
    return result


def predict_proba(model_data: Dict, from_source: List[str]) -> List[str]:
    """

    Args:
        model_data:
        from_source:

    Returns:

    """
    result = ['SELECT']

    sum_of_feats = ' + '.join(map(lambda x: f"({x[0]})*({x[1]['weight']})", model_data['features'].items()))
    result.append(f"    1/(1+EXP(-(({sum_of_feats})+({model_data['intercept']})))) AS result")

    result.append(f'FROM (')

    for s in from_source:
        result.append(f'    {s}')

    result.append(f')')
    return result


def get_sql_query(model_data: Dict, table_name: str) -> str:
    """

    Args:
        model_data:
        table_name:

    Returns:

    """
    buffer = StringIO()
    feats = preprocess_features(model_data, table_name)
    woe_features = transform_features(model_data, feats)
    result_sql = predict_proba(model_data, woe_features)

    for line in result_sql:
        buffer.write(line)
        buffer.write('\n')

    result = buffer.getvalue()
    buffer.close()
    return result
