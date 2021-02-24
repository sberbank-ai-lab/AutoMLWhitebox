from typing import Optional


def prepare_number(woe_dict, name: str, r_val: int = 3, round_features: int = 5,
                   nan_pattern: str = "({0} IS NULL OR {0} = 'NaN')",
                   preprocessing: Optional[str] = None):
    """
    Get encoding case when for number

    Args:
        woe_dict:
        name:
        r_val:
        round_features:
        nan_pattern:
        preprocessing:

    Returns:

    """
    # value in case
    f_val = name
    if preprocessing is not None:
        f_val = preprocessing.format(name)

    # search for NaN encoding
    for grp in woe_dict.cod_dict:
        if type(grp) is str and grp.startswith('__NaN_'):
            nan_val = round(woe_dict.cod_dict[grp], r_val)
            break
    else:
        raise ValueError('NaN encoding value does not exists in woe_dict')

    nan_case = nan_pattern.format(f_val)
    feature = """CASE\n  WHEN {0} THEN {1}\n""".format(nan_case, nan_val)
    # create regular bins
    for grp, val in enumerate(woe_dict.split):
        enc_val = round(woe_dict.cod_dict[grp], r_val)
        feature += """  WHEN {0} <= {1} THEN {2}\n""".format(f_val, round(val, round_features), enc_val)
    # create last else val
    enc_val = round(woe_dict.cod_dict[len(woe_dict.split)], r_val)
    feature += """  ELSE {1}\nEND AS {0}""".format(name, enc_val, )

    return feature


def check_cat_symb(x: str):
    """

    Args:
        x:

    Returns:

    """
    if type(x) is str:
        x = "'{0}'".format(x)
    else:
        x = str(x)

    return x


def prepare_category(woe_dict, name: str, r_val: int = 3,
                     nan_pattern: str = "({0} IS NULL OR LOWER(CAST({0} AS VARCHAR(50))) = 'nan')",
                     preprocessing: Optional[str] = None):
    """
    Get encoding case when for category

    Args:
        woe_dict:
        name:
        r_val:
        nan_pattern:
        preprocessing:

    Returns:

    """

    # value in case
    f_val = name
    if preprocessing is not None:
        f_val = preprocessing.format(name)

    # search for NaN and Small encodings
    nan_val, small_val, small_grp = None, None, None
    for grp in woe_dict.split:
        if type(grp) is str:
            if grp.startswith('__NaN_'):
                nan_grp = woe_dict.split[grp]
                nan_val = round(woe_dict.cod_dict[nan_grp], r_val)

            if grp.startswith('__Small_'):
                small_grp = woe_dict.split[grp]
                small_val = round(woe_dict.cod_dict[small_grp], r_val)

    # search for small in cod_dict
    for grp in woe_dict.cod_dict:
        if type(grp) is str:
            if grp.startswith('__NaN_'):
                nan_val = round(woe_dict.cod_dict[grp], r_val)
            if grp.startswith('__Small_'):
                small_val = round(woe_dict.cod_dict[grp], r_val)
                small_grp = -1

    assert nan_val is not None, 'NaN encoding value does not exists in woe_dict'
    assert small_val is not None, 'Small encoding value does not exists in woe_dict'

    feature = """CASE\n"""
    if nan_val != small_val:
        nan_case = nan_pattern.format(f_val)
        feature += """  WHEN {0} THEN {1}\n""".format(nan_case, nan_val)

    # create regular bins
    passed = {small_grp}
    for grp in woe_dict.split.values():
        if grp not in passed:

            search_vals = [x for x in woe_dict.split if woe_dict.split[x] == grp
                           and x not in {"__NaN_0__", "__NaN_maxfreq__", "__NaN_maxp__",
                                         "__NaN_minp__", "__Small_0__", "__Small_maxfreq__",
                                         "__Small_maxp__", "__Small_minp__", "__NaN__", "__Small__"}]
            length = len(search_vals)
            search_vals = list(map(check_cat_symb, search_vals))

            # filter NaN and Small cases separatly
            enc_val = round(woe_dict.cod_dict[grp], r_val)
            if length > 1:
                feature += """  WHEN {0} IN ({1}) THEN {2}\n""".format(f_val, ', '.join(search_vals), enc_val)
            elif length == 1:
                feature += """  WHEN {0} = {1} THEN {2}\n""".format(f_val, search_vals[0], enc_val)

            passed.add(grp)

    # create last ELSE with small
    feature += """  ELSE {1}\nEND AS {0}""".format(name, small_val, )

    return feature


def set_indent(x: str, n: int = 2):
    """

    Args:
        x:
        n:

    Returns:

    """
    indent = ' ' * n

    x = indent + x
    x = x.replace('\n', '\n' + indent)

    return x


def get_encoded_table(model, table_name, round_woe=3, round_features=5,
                      nan_pattern_numbers="({0} IS NULL OR {0} = 'NaN')",
                      nan_pattern_category="({0} IS NULL OR LOWER(CAST({0} AS VARCHAR(50))) = 'nan')",
                      preprocessing=None):
    """
    Get encoding table

    Args:
        model:
        table_name:
        round_woe:
        round_features:
        nan_pattern_numbers:
        nan_pattern_category:
        preprocessing:

    Returns:

    """
    if preprocessing is None:
        preprocessing = {}

    query = """SELECT\n"""

    for n, name in enumerate(model.features_fit.index):

        woe_dict = model.woe_dict[name]

        prep = None
        if name in preprocessing:
            prep = preprocessing[name]

        if woe_dict.f_type == 'cat':
            feature = prepare_category(woe_dict, name, round_woe, nan_pattern_category, prep)
        else:
            feature = prepare_number(woe_dict, name, round_woe, round_features, nan_pattern_numbers, prep)

        query += set_indent(feature)

        if (n + 1) != len(model.features_fit):
            query += ','

        query += '\n'

    query += """FROM {0}""".format(table_name)

    return query


def get_weights_query(model, table_name, output_name='PROB', alias='WOE_TAB', bypass_encoded=False, round_wts=3):
    """
    Calc prob over woe table

    Args:
        model:
        table_name:
        output_name:
        alias:
        bypass_encoded:
        round_wts:

    Returns:

    """
    query = """SELECT\n  1 / (1 + EXP(-({0}\n  ))) as {3}{1}\nFROM {2} as {4}"""

    dot = '\n    {0}'.format(round(model.intercept, round_wts))

    for name, val in zip(model.features_fit.index, model.features_fit.values):
        sign = '' if val < 0 else '+'
        dot += """\n    {0}{1}*{3}.{2}""".format(sign, round(val, round_wts), name, alias)

    other = ''
    if bypass_encoded:
        other = """,\n  {0}.*""".format(alias)

    return query.format(dot, other, table_name, output_name, alias)


def get_sql_inference_query(model, table_name, round_digits=3, round_features=5, output_name='PROB', alias='WOE_TAB',
                            bypass_encoded=True, template=None,
                            nan_pattern_numbers="({0} IS NULL OR {0} = 'NaN')",
                            nan_pattern_category="({0} IS NULL OR LOWER(CAST({0} AS VARCHAR(50))) = 'nan')",
                            preprocessing=None):
    """
    Get sql query

    Args:
        model:
        table_name:
        round_digits:
        round_features:
        output_name:
        alias:
        bypass_encoded:
        template:
        nan_pattern_numbers:
        nan_pattern_category:
        preprocessing:

    Returns:

    """
    assert template in ['td'] or template is None, 'Unknown template'

    if template == 'td':
        nan_pattern_numbers = '{0} IS NULL'
        nan_pattern_category = '{0} IS NULL'

    # get table with features
    encode_table = "({0})".format(get_encoded_table(model, table_name, round_digits, round_features,
                                                    nan_pattern_numbers, nan_pattern_category, preprocessing))
    encode_table = """\n  """ + set_indent(encode_table)

    # get table with weights
    query = get_weights_query(model, encode_table, output_name=output_name,
                              bypass_encoded=bypass_encoded, alias=alias, round_wts=round_digits)

    return query
