import os
from datetime import datetime
from shutil import copyfile

from jinja2 import FileSystemLoader, Environment

from ..logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """

    """

    def __init__(self):
        self.env = Environment(loader=FileSystemLoader(searchpath=os.path.dirname(__file__)))
        self.base_template = self.env.get_template('autowoe_report_template.html')

    def write_report_to_file(self, report_params):
        """

        Args:
            report_params:

        Returns:

        """
        with open(os.path.join(report_params['output_path'], 'autowoe_report.html'), "w") as f:
            f.write(self.base_template.render(
                report_name=str(report_params['report_name']),
                report_version=str(report_params['report_version_id']),
                city=str(report_params['city']),
                year=str(datetime.now().year),
                model_aim=str(report_params['model_aim']),
                model_name=str(report_params['model_name']),
                zakazchik=str(report_params['zakazchik']),
                high_level_department=str(report_params['high_level_department']),
                ds_name=str(report_params['ds_name']),
                target_descr=str(report_params["target_descr"]),
                non_target_descr=str(report_params["non_target_descr"]),

                count_train=report_params["count_train"],
                train_target_cnt=report_params["train_target_cnt"],
                train_nontarget_cnt=report_params["train_nontarget_cnt"],
                train_target_perc=report_params["train_target_perc"],
                train_auc_full=report_params["train_auc_full"],
                train_gini_full=report_params["train_gini_full"],

                count_test=report_params["count_test"],
                test_target_cnt=report_params["test_target_cnt"],
                test_nontarget_cnt=report_params["test_nontarget_cnt"],
                test_target_perc=report_params["test_target_perc"],
                test_auc_full=report_params["test_auc_full"],
                test_gini_full=report_params["test_gini_full"],
                train_gini_confint=report_params["train_gini_confint"],
                test_gini_confint=report_params["test_gini_confint"],

                model_coef=report_params["model_coef"],
                p_vals=report_params["p_vals"],
                p_vals_test=report_params["p_vals_test"],

                final_nan_stat=report_params["final_nan_stat"],

                features_roc_auc=report_params["features_roc_auc"],

                features_woe=report_params["features_woe"],
                woe_bars=report_params["woe_bars"],
                backlash_plots=report_params["backlash_plots"],
                train_vif=report_params["train_vif"],
                psi_total=report_params["psi_total"],
                psi_zeros=report_params["psi_zeros"],
                psi_ones=report_params["psi_ones"],
                psi_binned_total=report_params["psi_binned_total"],
                psi_binned_zeros=report_params["psi_binned_zeros"],
                psi_binned_ones=report_params["psi_binned_ones"],

                scorecard=report_params["scorecard"],
                feature_history=report_params["feature_history"],
                feature_contribution=report_params["feature_contribution"],
                corr_map_table=report_params["corr_map_table"],
                binned_p_stats_train=report_params["binned_p_stats_train"],
                binned_p_stats_test=report_params["binned_p_stats_test"],

                dategrouped_value=report_params["dategrouped_value"],
                dategrouped_gini=report_params["dategrouped_gini"],
                dategrouped_nan=report_params["dategrouped_nan"],
            ))

    def generate_report(self, report_params):
        """

        Args:
            report_params:

        Returns:

        """
        copyfile(os.path.join(os.path.dirname(__file__), 'shaptxt'),
                 os.path.join(report_params['output_path'], 'shap.js'))

        self.write_report_to_file(report_params)

        logger.info('Successfully wrote {}.'.format(os.path.join(report_params['output_path'], 'autowoe_report.html')))
