import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, roc_curve


def plot_roc_curve_image(y_true, y_pred, path):
    sns.set(style="whitegrid", font_scale=1.5)
    plt.figure(figsize=(10, 10))

    fpr_reg, tpr_reg, _ = roc_curve(y_true, y_pred)
    auc_score_reg = roc_auc_score(y_true, y_score=y_pred)

    lw = 2
    plt.plot(fpr_reg, tpr_reg, color='darkorange',
             lw=lw, label='WhiteBox модель (GINI = {:.3f})'.format(2 * auc_score_reg - 1))
    plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--', label='Случайная модель')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    lgd = plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    plt.xticks(np.arange(0, 1.01, 0.05), rotation=45)
    plt.yticks(np.arange(0, 1.01, 0.05))
    plt.grid(color='gray', linestyle='-', linewidth=1)
    plt.title('ROC кривая (GINI = {:.3f})'.format(2 * auc_score_reg - 1))
    plt.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

def plot_double_roc_curve(train_y_true, train_y_pred, test_y_true, test_y_pred, path):
    sns.set(style="whitegrid", font_scale=1.5)
    plt.figure(figsize=(10, 10))

    train_fpr_reg, train_tpr_reg, _ = roc_curve(train_y_true, train_y_pred)
    train_auc_score_reg = roc_auc_score(train_y_true, y_score=train_y_pred)
    test_fpr_reg, test_tpr_reg, _ = roc_curve(test_y_true, test_y_pred)
    test_auc_score_reg = roc_auc_score(test_y_true, y_score=test_y_pred)

    lw = 2
    plt.plot(
        train_fpr_reg,
        train_tpr_reg,
        color='darkorange',
        lw=lw,
        label='По данным train (GINI = {:.3f})'.format(2 * train_auc_score_reg - 1)
    )
    plt.plot(
        test_fpr_reg,
        test_tpr_reg,
        color='blue',
        lw=lw,
        label='По данным test (GINI = {:.3f})'.format(2 * test_auc_score_reg - 1)
    )
    plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--', label='Случайная модель')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.xticks(np.arange(0, 1.01, 0.05), rotation=45)
    plt.yticks(np.arange(0, 1.01, 0.05))
    plt.grid(color='gray', linestyle='-', linewidth=1)
    plt.title('ROC кривая')
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def plot_roc_curve_feature_image(feature_name, y_true, y_pred, path):
    sns.set(style="whitegrid", font_scale=1.5)
    plt.figure(figsize=(10, 10))

    fpr_reg, tpr_reg, _ = roc_curve(y_true, y_pred)
    auc_score_reg = roc_auc_score(y_true, y_score=y_pred)

    lw = 2
    plt.plot(fpr_reg, tpr_reg, color='darkorange',
             lw=lw, label=feature_name + ' (GINI = {:.3f})'.format(2 * auc_score_reg - 1))
    plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--', label='Случайная модель')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    lgd = plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    plt.xticks(np.arange(0, 1.01, 0.05), rotation=45)
    plt.yticks(np.arange(0, 1.01, 0.05))
    plt.grid(color='gray', linestyle='-', linewidth=1)
    plt.title('ROC кривая(GINI = {:.3f})'.format(2 * auc_score_reg - 1) + f" для признака {feature_name}")
    plt.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def plot_model_weights(features, path):
    sns.set(style="whitegrid", font_scale=1.5)
    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(features.index, features.values, color='g')
    lgd = plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    plt.title('Параметры модели', fontsize=28)
    plt.xlabel('Признаки', fontsize=20)
    plt.ylabel('Значения коэффициентов регрессии', fontsize=20)
    plt.xticks(fontsize=15, rotation=90)
    plt.yticks(fontsize=15)
    plt.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def plot_feature_split(feature_name, features, path):
    sns.set(style="whitegrid", font_scale=1.5)
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(features.index, features.values, color='g')
    lgd = plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
    plt.title("Разбиение признака " + feature_name + " на бины и WoE значения в них")
    plt.xlabel('Бины', fontsize=20)
    plt.ylabel('WoE значения', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

def plot_ginis(data_enc, target, path):
    sns.set(style="whitegrid", font_scale=1.5)
    feats = list(data_enc.columns)
    aucs = [roc_auc_score(target, -data_enc[col].values) for col in feats]
    ginis = [(x - 0.5) * 2 for x in aucs]
    ginis = pd.Series(ginis, index=feats).sort_values(ascending=True)
    pl = ginis.plot(kind='barh', figsize=(10, 10))
    pl.get_figure().savefig(path, bbox_inches='tight')
    plt.close()

def plot_woe_bars(train_enc, train_target, test_enc, test_target, target_name, column, path):
    sns.set(style="whitegrid", font_scale=1.5)
    names = ['train', 'test']
    samples = []
    for df, target in zip([train_enc, test_enc], [train_target, test_target]):
        df_copy = df.copy().round(3)
        df_copy[target_name] = target
        samples.append(df_copy)
    
    samples = [x[[target_name, column]].groupby(column)[target_name].agg(['mean', 'count']).reset_index()
               for x in samples]
    
    for df in samples:
        df['count'] /= df['count'].sum()
        
        df.rename({'count': 'Freq', 'mean': 'DefaultRate',
                               column: 'WOE: ' + column}, inplace=True, axis=1)
        
    total = pd.concat(samples, axis=0, ignore_index=True)
    total['_sample_'] = np.concatenate([[n] * x.shape[0] for (n, x) in zip(names, samples)])
    
    plt.figure(figsize=(10, 10))
    ax = sns.barplot(x='WOE: '+column, hue="_sample_", y="Freq", data=total, palette=sns.color_palette("Accent", 7))
    ax2 = ax.twinx()
    pl = sns.pointplot(x='WOE: '+column, y="DefaultRate", hue="_sample_", kind="point", data=total, ax=ax2)
    pl.get_figure().savefig(path, bbox_inches='tight')
    plt.close()

def plot_backlash_check(predict_proba, data_enc, target, col, path):
    sns.set(style="whitegrid", font_scale=1.5)
    df = pd.DataFrame({'pred': predict_proba, col: data_enc[col], 'Target': target})
    grp = df.groupby(col)['pred', 'Target'].mean()
    grp.plot(figsize=(10, 10)).get_figure().savefig(path, bbox_inches='tight')
    plt.close()

def plot_binned(data_binned, path1, path2):
    sns.set(style="whitegrid", font_scale=1.5)
    pl = (data_binned.groupby('ScoreBin').size().sort_index() / data_binned.shape[0]).plot(figsize=(10,10), kind='bar')
    pl.get_figure().savefig(path1, bbox_inches='tight')
    plt.close()

    neg = (data_binned[data_binned['Target']==0].groupby('ScoreBin').size().sort_index() / 
     (data_binned.shape[0] - data_binned['Target'].sum()))# .plot(kind='bar', cmap='Accent')

    pos = (data_binned[data_binned['Target']==1].groupby('ScoreBin').size().sort_index() / 
     (data_binned['Target'].sum()))# .plot(kind='bar', cmap='Accent', color='blue')

    pl = pd.DataFrame({'positive': pos, 'negative': neg}).plot(figsize=(10,10), kind='bar', cmap='Accent', width=1)
    pl.get_figure().savefig(path2, bbox_inches='tight')
    plt.close()

def plot_binned_stats(data_binned, path):
    sns.set(style="whitegrid", font_scale=1.5)
    pl = data_binned[['ScoreBin', 'P']].boxplot(by='ScoreBin', rot=90, figsize=(10, 10))
    pl.get_figure().savefig(path, bbox_inches='tight')
    plt.close()

def plot_corr_heatmap(corr_map, path):
    sns.set(style="whitegrid", font_scale=1.5)
    plt.figure(figsize=(20, 10))
    pl = sns.heatmap(corr_map, annot = True, annot_kws={"size": 8}, fmt='.1g')
    pl.get_figure().savefig(path, bbox_inches='tight')
    plt.close()

def plot_mean_target(train_binned, test_binned, path):
    sns.set(style="whitegrid", font_scale=1.5)
    train_stat = train_binned.groupby('ScoreBin').agg(mean_target=('Target', 'mean'))
    test_stat = test_binned.groupby('ScoreBin').agg(mean_target=('Target', 'mean'))
    df = pd.DataFrame({'train_mean_target': train_stat['mean_target'], 'test_mean_target': test_stat['mean_target']})
    pl = df.plot(figsize=(10,10), kind='bar', cmap='Accent', width=1)
    pl.get_figure().savefig(path, bbox_inches='tight')
    plt.close()
