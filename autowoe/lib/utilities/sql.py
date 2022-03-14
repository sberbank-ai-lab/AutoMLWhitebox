"""SQL-query utilities."""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from autowoe.lib.pipelines.pipeline_feature_special_values import MARK_SET
from autowoe.lib.pipelines.pipeline_feature_special_values import NAN_SET
from autowoe.lib.pipelines.pipeline_feature_special_values import SMALL_SET
from autowoe.lib.pipelines.pipeline_feature_special_values import is_mark_prefix
from autowoe.lib.utilities.utils import TaskType
from autowoe.lib.woe.woe import WoE


def prepare_number(
    woe_dict: WoE,
    name: str,
    r_val: int = 3,
    round_features: int = 5,
    nan_pattern: str = "({0} IS NULL OR {0} = 'NaN')",
    preprocessing: Optional[str] = None,
    mark_values: Optional[Dict[str, Tuple[Any]]] = None,
    mark_encoding: Optional[Dict[Any, str]] = None,
):
    """Get encoding case when for number.

    Args:
        woe_dict: Dictonary of WoE values.
        name: Name of feature.
        r_val: Numbers after the decimal point.
        round_features: Numbers after the decimal point.
        nan_pattern: Expression for nan processing.
        preprocessing: Name preprocessing.
        mark_values: List of marked values.
        mark_encoding: Map marked value to code.

    Returns:
        sql query part for number.

    """
    # value in case
    feature_mark_values = [] if mark_values is None else mark_values.get(name, [])

    f_val = name
    if preprocessing is not None:
        f_val = preprocessing.format(name)

    # search for NaN encoding
    for grp in woe_dict.cod_dict:
        if type(grp) is str and grp.startswith("__NaN_"):
            nan_val = round(woe_dict.cod_dict[grp], r_val)
            break
    else:
        raise ValueError("NaN encoding value does not exists in woe_dict")

    nan_case = nan_pattern.format(f_val)
    feature = """CASE\n  WHEN {0} THEN {1}\n""".format(nan_case, nan_val)

    # if feature_mark_values is not None:
    #     for grp in woe_dict.cod_dict:
    #         if type(grp) is str and grp.startswith("__Mark_"):
    #             mark_val = round(woe_dict.cod_dict[grp], r_val)
    #             break

    #     mark_case = ", ".join(str(m) for m in feature_mark_values)
    #     feature += """  WHEN {} IN ({}) THEN {}\n""".format(f_val, mark_case, mark_val)

    # create regular bins
    for grp, val in enumerate(woe_dict.split):
        enc_val = round(woe_dict.cod_dict[grp], r_val)
        feature += """  WHEN {0} <= {1} THEN {2}\n""".format(f_val, round(val, round_features), enc_val)

    for mv in feature_mark_values:
        # enc = "__Mark__{}__".format(mv)
        enc = mark_encoding[name][mv]
        enc_val = round(woe_dict.cod_dict[enc], r_val)
        feature += """  WHEN {0} == {1} THEN {2}\n""".format(f_val, mv, enc_val)

    # create last else val
    enc_val = round(woe_dict.cod_dict[len(woe_dict.split)], r_val)
    feature += """  ELSE {1}\nEND AS {0}""".format(
        name,
        enc_val,
    )

    return feature


def check_cat_symb(x: Union[str, Any]) -> str:
    """Wrap to quotes.

    Args:
        x: Value.

    Returns:
        quoted string.

    """
    if type(x) is str:
        x = "'{0}'".format(x)
    else:
        x = str(x)

    return x


def prepare_category(
    woe_dict,
    name: str,
    r_val: int = 3,
    nan_pattern: str = "({0} IS NULL OR LOWER(CAST({0} AS VARCHAR(50))) = 'nan')",
    preprocessing: Optional[str] = None,
    mark_values: Optional[Dict[str, List[Any]]] = None,
    mark_encoding: Optional[Dict[Any, str]] = None,
):
    """Get encoding case when for category.

    Args:
        woe_dict: Dictonary of WoE values.
        name: Name of feature.
        r_val: Numbers after the decimal point.
        nan_pattern: Expression for nan processing.
        preprocessing: Name preprocessing.
        mark_values: List of mark values.
        mark_encoding: Map marked value to code.

    Returns:
        sql query part for category.

    """
    feature_mark_values = [] if mark_values is None else mark_values.get(name, [])

    # value in case
    f_val = name
    if preprocessing is not None:
        f_val = preprocessing.format(name)

    # search for Mark, NaN and Small encodings
    nan_val, small_val, small_grp = None, None, None
    for grp in woe_dict.split:
        if type(grp) is str:
            if grp.startswith("__NaN_"):
                nan_grp = woe_dict.split[grp]
                nan_val = round(woe_dict.cod_dict[nan_grp], r_val)

            if grp.startswith("__Small_"):
                small_grp = woe_dict.split[grp]
                small_val = round(woe_dict.cod_dict[small_grp], r_val)

            # if grp.startswith("__Mark_"):
            #     mark_grp = woe_dict.split[grp]
            #     mark_val = round(woe_dict.cod_dict[mark_grp], r_val)

    # search for small in cod_dict
    for grp in woe_dict.cod_dict:
        if type(grp) is str:
            if grp.startswith("__NaN_"):
                nan_val = round(woe_dict.cod_dict[grp], r_val)
            if grp.startswith("__Small_"):
                small_val = round(woe_dict.cod_dict[grp], r_val)
                small_grp = -1

    assert nan_val is not None, "NaN encoding value does not exists in woe_dict"
    # assert small_val is not None, "Small encoding value does not exists in woe_dict"
    # TODO: assert for mark val

    feature = """CASE\n"""
    if nan_val != small_val:
        nan_case = nan_pattern.format(f_val)
        feature += """  WHEN {0} THEN {1}\n""".format(nan_case, nan_val)

    # if feature_mark_values is not None:
    #     mark_case = []
    #     for m in feature_mark_values:
    #         if isinstance(m, str):
    #             fmt = "'{}'".format(m)
    #         else:
    #             fmt = str(m)
    #         mark_case.append(fmt)
    #     mark_case = ", ".join(mark_case)
    #     feature += """  WHEN {} IN ({}) THEN {}\n""".format(f_val, mark_case, mark_val)

    # create regular bins
    passed = {small_grp}
    for grp in woe_dict.split.values():
        if grp not in passed:

            search_vals = [
                x
                for x in woe_dict.split
                if woe_dict.split[x] == grp and x not in {*SMALL_SET, *NAN_SET, *MARK_SET} and not is_mark_prefix(x)
            ]
            length = len(search_vals)
            search_vals = list(map(check_cat_symb, search_vals))

            # filter NaN and Small cases separatly
            enc_val = round(woe_dict.cod_dict[grp], r_val)
            if length > 1:
                feature += """  WHEN {0} IN ({1}) THEN {2}\n""".format(f_val, ", ".join(search_vals), enc_val)
            elif length == 1:
                feature += """  WHEN {0} == {1} THEN {2}\n""".format(f_val, search_vals[0], enc_val)

            passed.add(grp)

    for mv in feature_mark_values:
        # enc = "__Mark__{}__".format(mv)
        enc = mark_encoding[name][mv]
        idx_enc = woe_dict.split[enc]
        enc_val = round(woe_dict.cod_dict[idx_enc], r_val)
        feature += """  WHEN {0} == {1} THEN {2}\n""".format(f_val, check_cat_symb(mv), enc_val)

    # create last ELSE with small
    feature += """  ELSE {1}\nEND AS {0}""".format(
        name,
        small_val,
    )

    return feature


def set_indent(x: str, n: int = 2):
    """Indentation in spaces for a line.

    Args:
        x: String.
        n: Number of spaces.

    Returns:
        Shifted string.

    """
    indent = " " * n

    x = indent + x
    x = x.replace("\n", "\n" + indent)

    return x


def get_encoded_table(
    model,
    table_name,
    round_woe=3,
    round_features=5,
    nan_pattern_numbers="({0} IS NULL OR {0} = 'NaN')",
    nan_pattern_category="({0} IS NULL OR LOWER(CAST({0} AS VARCHAR(50))) = 'nan')",
    preprocessing=None,
    mark_values=None,
    mark_encoding=None,
):
    """Get encoding table.

    Args:
        model: Model.
        table_name: Feature table name.
        round_woe: Numbers after the decimal point.
        round_features: Numbers after the decimal point.
        nan_pattern_numbers: Expression for nan processing in number feature.
        nan_pattern_category: Expression for nan processing in category feature.
        preprocessing: Name processing.
        mark_values: List of mark values.
        mark_encoding: Map marked value to code.

    Returns:
        query.

    """
    if preprocessing is None:
        preprocessing = {}

    query = """SELECT\n"""

    for n, name in enumerate(model.features_fit.index):

        woe_dict = model.woe_dict[name]

        prep = None
        if name in preprocessing:
            prep = preprocessing[name]

        if woe_dict.f_type == "cat":
            feature = prepare_category(
                woe_dict, name, round_woe, nan_pattern_category, prep, mark_values, mark_encoding
            )
        else:
            feature = prepare_number(
                woe_dict, name, round_woe, round_features, nan_pattern_numbers, prep, mark_values, mark_encoding
            )

        query += set_indent(feature)

        if (n + 1) != len(model.features_fit):
            query += ","

        query += "\n"

    query += """FROM {0}""".format(table_name)

    return query


def get_weights_query(model, table_name, output_name="PROB", alias="WOE_TAB", bypass_encoded=False, round_wts=3):
    """Calc prob over woe table.

    Args:
        model: Model.
        table_name: WoE table name.
        output_name: Output name.
        alias: Alias.
        bypass_encoded: Add encoded features to result query.
        round_wts: Round.

    Returns:
        query.

    """
    if model.params["task"] == TaskType.BIN:
        # query = """SELECT\n  1 / (1 + EXP(-({0}\n  ))) as {3}{1}\nFROM {2} as {4}"""
        query = """SELECT\n  1 / (1 + EXP(-({LIN_FUN}\n  ))) as {OUTPUT_NAME}{WOE_VALS}\nFROM {TABLE_NAME} as {ALIAS}"""
    else:
        # query = """SELECT\n ( {0}\n ) as {3}{1}\nFROM {2} as {4}"""
        query = """SELECT\n ( {S} * ( {LIN_FUN}\n) + {M}\n ) as {OUTPUT_NAME}{WOE_VALS}\nFROM {TABLE_NAME} as {ALIAS}"""

    dot = "\n    {0}".format(round(model.intercept, round_wts))

    for name, val in zip(model.features_fit.index, model.features_fit.values):
        sign = "" if val < 0 else "+"
        dot += """\n    {0}{1}*{3}.{2}""".format(sign, round(val, round_wts), name, alias)

    other = ""
    if bypass_encoded:
        other = """,\n  {0}.*""".format(alias)

    # return query.format(dot, other, table_name, output_name, alias)
    query_args = {
        "LIN_FUN": dot,
        "WOE_VALS": other,
        "TABLE_NAME": table_name,
        "OUTPUT_NAME": output_name,
        "ALIAS": alias,
    }
    if model.params["task"] == TaskType.REG:
        query_args["S"] = round(model._target_std, round_wts)
        query_args["M"] = round(model._target_mean, round_wts)

    return query.format(**query_args)


def get_sql_inference_query(
    model,
    table_name,
    round_digits=3,
    round_features=5,
    output_name="PROB",
    alias="WOE_TAB",
    bypass_encoded=True,
    template=None,
    nan_pattern_numbers="({0} IS NULL OR {0} = 'NaN')",
    nan_pattern_category="({0} IS NULL OR LOWER(CAST({0} AS VARCHAR(50))) = 'nan')",
    preprocessing=None,
    mark_values=None,
    mark_encoding=None,
):
    """Get sql query.

    Args:
        model: Model.
        table_name: Feature table name.
        round_digits: Round digits.
        round_features: Round digits of features.
        output_name: Output name.
        alias: Alias.
        bypass_encoded: Add encoded features to result query.
        template: T.
        nan_pattern_numbers: Expression for nan processing in number feature.
        nan_pattern_category: Expression for nan processing in category feature.
        preprocessing: Name preprocessing.
        mark_values: List of marked values.
        mark_encoding: Map marked value to code.

    Returns:
        query.

    """
    assert template in ["td"] or template is None, "Unknown template"

    if template == "td":
        nan_pattern_numbers = "{0} IS NULL"
        nan_pattern_category = "{0} IS NULL"

    # get table with features
    encode_table = "({0})".format(
        get_encoded_table(
            model,
            table_name,
            round_digits,
            round_features,
            nan_pattern_numbers,
            nan_pattern_category,
            preprocessing,
            mark_values,
            mark_encoding,
        )
    )
    encode_table = """\n  """ + set_indent(encode_table)

    # get table with weights
    query = get_weights_query(
        model, encode_table, output_name=output_name, bypass_encoded=bypass_encoded, alias=alias, round_wts=round_digits
    )

    return query
