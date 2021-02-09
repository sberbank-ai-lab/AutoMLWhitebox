from typing import Dict, List
from io import StringIO
from collections import defaultdict


def prepare_volatile_tables(model_data, table_name):
    result = []
    create_table_cmd = 'CREATE VOLATILE TABLE {} (Name VARCHAR(50),Value DECIMAL(12,6)) ON COMMIT PRESERVE ROWS;'
    temp_tables = []
    for i, feature in enumerate(model_data['features']):
        woe = model_data['features'][feature]
        nan_value = woe['nan_value']
        if woe['f_type'] != 'real':
            create_table = create_table_cmd.format(feature)
            result.append(create_table)
            temp_tables.append(feature)
            for cat, woe_idx in woe['cat_map'].items():
                result.append(f"INSERT INTO {feature} VALUES ('{cat}',{woe['cod_dict'][woe_idx]});")
            result.append('')
    return result, temp_tables

def get_teradata_features(model_data, table_name, temp_tables):
    result = ['SELECT']

    for i, feature in enumerate(model_data['features']):
        woe = model_data['features'][feature]
        nan_value = woe['nan_value']

        if woe['f_type'] == 'real':
            map_str = ''
            if len(woe['splits']) > 0:
                splits = [(woe['splits'][i], woe['cod_dict'][i]) for i in range(len(woe['splits']))]
                map_str = ' '.join([f'WHEN {table_name}.{feature} <= {x[0]} THEN {x[1]}' for x in splits])

            nan_case = woe['spec_cod'][nan_value] if nan_value in woe['spec_cod'] else 0.0
            else_case = woe['cod_dict'][len(woe['splits'])]
            if map_str:
                map_str += ' '
            string = f"CASE WHEN {table_name}.{feature} IS NULL THEN {nan_case} {map_str}ELSE {else_case} END"
        else:
            nan_case = 0.0
            if nan_value in woe['spec_cod']:
                nan_case = woe['spec_cod'][nan_value]
            elif nan_value in woe['cat_map']:
                nan_case = woe['cod_dict'][woe['cat_map'][nan_value]]

            valid_cats = [cat_name for cat_name in woe['cat_map'] if cat_name != nan_value and cat_name not in woe['spec_cod']]
            if len(valid_cats) > 0:
                small_case = nan_case
                if 'spec_cat' in woe:
                    _, spec_value = woe['spec_cat']
                    if spec_value in woe['spec_cod']:
                        small_case = woe['spec_cod'][spec_value]
                    elif spec_value in woe['cat_map']:
                        small_case = woe['cod_dict'][woe['cat_map'][spec_value]]
            else:
                small_case = f"{woe['cod_dict'][0]}"

            string = f"CASE WHEN {table_name}.{feature} IS NULL THEN {nan_case} " + \
                f"WHEN {feature}.Value IS NULL THEN {small_case} " + \
                f"ELSE {feature}.Value END"

        is_not_final = i != len(model_data['features']) - 1
        result.append(f"    ({string}) AS {feature}" + (',' if is_not_final else ''))

    result.append(f'FROM {table_name}')
    for temp_table in temp_tables:
        result.append(f'LEFT JOIN {temp_table} ON {table_name}.{temp_table} = {temp_table}.Name')

    return result


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
            string = f"(CASE WHEN {feature} IS NULL OR LOWER(CAST({feature} AS {str_type})) = 'nan' " + \
f"THEN NULL WHEN {feature} NOT IN ({big_cats_str}) THEN '{spec_value}' ELSE CAST({feature} AS {str_type}) END) AS {feature}"

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
            map_str = ''
            if len(woe['splits']) > 0:
                splits = [(woe['splits'][i], woe['cod_dict'][i]) for i in range(len(woe['splits']))]
                map_str = ' '.join([f'WHEN {feature} <= {x[0]} THEN {x[1]}' for x in splits])

            nan_case = woe['spec_cod'][nan_value] if nan_value in woe['spec_cod'] else 0.0
            else_case = woe['cod_dict'][len(woe['splits'])]
            if map_str:
                map_str += ' '
            string = f"CASE WHEN {feature} IS NULL THEN {nan_case} {map_str}ELSE {else_case} END"
        else:
            if nan_value in woe['spec_cod']:
                nan_case = woe['spec_cod'][nan_value]
            elif nan_value in woe['cat_map']:
                nan_case = woe['cod_dict'][woe['cat_map'][nan_value]]
            else:
                nan_case = 0.0

            spec_case = ' '.join([f"WHEN {feature} = '{k}' THEN {v}" for k, v in woe['spec_cod'].items() if k != nan_value])
            
            cat_map = defaultdict(list)
            for cat, woe_idx in woe['cat_map'].items():
                if cat != nan_value and cat not in woe['spec_cod']:
                    cat_map[woe_idx].append(cat)
            
            cat_map = {woe_idx: ','.join(map(lambda x: f"'{x}'", cat_list)) for woe_idx, cat_list in cat_map.items()}

            map_str = ' '.join([f"WHEN {feature} IN ({cats}) THEN {woe['cod_dict'][woe_idx]}" for woe_idx, cats in cat_map.items()])

            if len(cat_map) > 0:
                else_case = nan_case
            else:
                else_case = f"{woe['cod_dict'][0]}"

            if spec_case:
                spec_case += ' '
            if map_str:
                map_str += ' '
            string = f"CASE WHEN {feature} IS NULL THEN {nan_case} {spec_case}{map_str}ELSE {else_case} END"

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

    result.append(f');')
    return result


def get_teradata_query(model_data: Dict, table_name: str) -> str:
    """
    Args:
        model_data:
        table_name:
    Returns:
    """
    buffer = StringIO()
    prep_cmds, temp_tables = prepare_volatile_tables(model_data, table_name)
    for line in prep_cmds:
        buffer.write(line)
        buffer.write('\n')
    
    woe_features = get_teradata_features(model_data, table_name, temp_tables)
    result_sql = predict_proba(model_data, woe_features)

    for line in result_sql:
        buffer.write(line)
        buffer.write('\n')

    result = buffer.getvalue()
    buffer.close()
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