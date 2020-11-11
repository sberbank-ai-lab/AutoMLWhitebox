## Sberbank version of AutoWoE


В данном репозитории находится библиотека **AutoWoE**, разработанная лабораторией AI. Библиотека используется для автоматической разработки интерпретируемых моделей (Биннинг + WoE + отбор признаков + логрегресиия)

**Авторы:** Антон Вахрушев, Григорий Пенкин

Установка 
pip install -r requirements.txt
python setup.py install 

Примеры использования описаны в `demonstration` ноутбукe

По всем вопросам/пожеланиям по работе данного решения просьба писать на почту:
- Антон Вахрушев (AGVakhrushev@sberbank.ru)



## Whitebox pipeline:

###  General params:

    - n_jobs
    - debug

### 0) Simple typing and trash removal
####    0.0) Remove trash feats
    
        Medium:
        - th_nan 
        - th_const 
        
####    0.1) Typing (auto and user defined)
        
        Critical:
        - features_type (dict) {'age': 'real', 'education': 'cat', 'birth_date': (None, ("d", "wd"), ...}
        
####    0.2) Dates and categories encoding
        
        Critical:
        - features_type (for datetimes)
        
        Optional:
        - cat_alpha (int) - greater means more conservative encoding
    
        
### 1) Initial feature selection (selection based on gbm importance)

    Critical:
    - select_type (None or int)
    - imp_type (if type(select_type) is int 'perm_imt'/'feature_imp') 
    
    Optional:
    - imt_th (float) - threshold for select_type is None
    
### 2) Binning:
    
    Critical:
    - monotonic / features_monotone_constraints 
    - max_bin_count / max_bin_count
    - min_bin_size
    
    - cat_merge_to
    - nan_merge_to
    
    Medium:
    - force_single_split
    
    Optional:
    - min_bin_mults
    - min_gains_to_split

### 3) WoE estimation WoE = LN( ((% 0 in bin) / (% 0 in sample)) / ((% 1 in bin) / (% 1 in sample)) ):
    
    Critical:
    - oof_woe
    
    Optional:
    - woe_diff_th
    - n_folds (if oof_woe)

### 4) Post selection:

####    4.0) Partial dependencies with target
    
    Critical:
    - auc_th
    
####    4.1) VIF 
    
    Critical:
    - vif_th
    
####    4.2) Partial correlcations
    
    Critical:
    - pearson_th
    
### 5) Model based selection
    
    Optional:
    - n_folds
    - l1_grid_size
    - l1_exp_scale


### 6) Final model refit:

    Critical:
    - regularized_refit
    - p_val (if not regularized_refit)
    - validation (if not regularized_refit)
    
    Optional:
    - interpreted_model
    - l1_grid_size (if regularized_refit)
    - l1_exp_scale (if regularized_refit)
    
### 7) Report generation 

    - report_params
