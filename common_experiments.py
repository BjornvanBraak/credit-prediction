#EXTERNAL PACKAGES
from aif360 import datasets
from aif360.algorithms.preprocessing import DisparateImpactRemover

#INTERNAL PACKAGES
from common_ml_pipeline import build_grid_search

def adjust_naming_catboost(param_grid_experiment_class_weight):
    copy = param_grid_experiment_class_weight.copy()
    copy['classifier__class_weights'] = copy['classifier__class_weight']
    del copy['classifier__class_weight']
    return copy

def adjust_naming_param_grid_for_catboost(param_grid):
    adjusted_param_grid = None
    if isinstance(param_grid, list):
        adjusted_param_grid = []
        for i in range(len(param_grid)):
            adjusted = adjust_naming_catboost(param_grid[i])
            adjusted_param_grid.append(adjusted)
    elif isinstance(param_grid, dict):
        adjusted_param_grid = adjust_naming_catboost(param_grid_experiment_class_weight)
    else:
        raise Exception("param_grid is not of a valid instance type " , param_grid)
    return adjusted_param_grid  

# EXPERIMENT 1
def perform_experiment_1(estimator, X, y, random_state, evaluation_metrics, all_class_weights, all_over_sampling_except_0, nominal_features, ordinal_features_dict, discrete_features, continuous_features, dataset_size_GB):
    param_grid_experiment_class_weight = [
    #zero case do not apply oversampling (the smotenc throwed errors when applying the original ratio
    #the most likely reason for this is that the stratifiedKFoldRatio was a little of from the original ratio
    {
      'oversample': ['passthrough'],
      'undersample': ['passthrough'], #skip undersampling
      'classifier__class_weight': all_class_weights
    },
    {
      'oversample__sampling_strategy': all_over_sampling_except_0,
      'undersample': ['passthrough'], #skip undersampling
      'classifier__class_weight': all_class_weights
    } 
    ]

    if estimator.__class__.__name__ == 'CatBoostClassifier':
        param_grid_experiment_class_weight = adjust_naming_param_grid_for_catboost(param_grid_experiment_class_weight) 
        
    grid_search = build_grid_search(
        estimator=estimator, 
        param_grid=param_grid_experiment_class_weight, 
        random_state=random_state, 
        evaluation_metrics=evaluation_metrics,
        nominal_features=nominal_features, 
        ordinal_features_dict=ordinal_features_dict, 
        discrete_features=discrete_features, 
        continuous_features=continuous_features,
        dataset_size_GB=dataset_size_GB
    )
    grid_search.fit(X,y)
    return grid_search

#EXPERIMENT 2
def perform_experiment_2(estimator, X, y, random_state, evaluation_metrics, all_undersampling_ratios, all_over_sampling_except_0, class_weight_none, nominal_features, ordinal_features_dict, discrete_features, continuous_features, dataset_size_GB):
    # print("oversampling_ratios: ", all_over_sampling_except_0)
    # print("undersampling_ratios: ", all_undersampling_ratios)

    param_grid_experiment_undersampling = [
    {
      'oversample': ['passthrough'],
      'undersample__sampling_strategy': all_undersampling_ratios, #skip undersampling
      'classifier__class_weight': [class_weight_none]
    },
    {
      'oversample__sampling_strategy': all_over_sampling_except_0,
      'undersample__sampling_strategy': all_undersampling_ratios, #skip undersampling
      'classifier__class_weight': [class_weight_none]
    }
    ]
    if estimator.__class__.__name__ == 'CatBoostClassifier':
        param_grid_experiment_undersampling = adjust_naming_param_grid_for_catboost(param_grid_experiment_undersampling) 
    #build grid_search
    grid_search = build_grid_search(
        estimator=estimator,
        param_grid=param_grid_experiment_undersampling, 
        random_state=random_state, 
        evaluation_metrics=evaluation_metrics,
        nominal_features=nominal_features, 
        ordinal_features_dict=ordinal_features_dict, 
        discrete_features=discrete_features, 
        continuous_features=continuous_features,
        dataset_size_GB=dataset_size_GB
    )
    grid_search.fit(X,y)
    return grid_search


#NOT SURE WHAT THIS DOES:
# #careful changing these, result visualisation depends on these
# global grid_searches 
# grid_searches = [
#   grid_search_experiment_1_rf, 
#   grid_search_experiment_2_rf, 
#   grid_search_experiment_1_lightgbm, 
#   grid_search_experiment_2_lightgbm, 
#   grid_search_experiment_1_catboost, 
#   grid_search_experiment_2_catboost
# ]


#experiment 3 - NEEDS WORK

# global repair_levels
# repair_levels = np.linspace(0., 1., 5)


# # import multiprocessing as mp
# # def SPD_score(estimator, X, y_true, bias_colname, unprivileged_value, privileged_value):
# #   y_pred = estimator.predict(X)
# #   if len(y_pred) != len(y_true):
# #     raise Exception("not equal length")
# #   bias_sensitive_arr = X[bias_predictor_name]
# #   SPD_abs = compute_abs_statitical_parity_difference(bias_colname, bias_sensitive_arr, unprivileged_value, privileged_value, y_true, y_pred)
# #   return SPD_abs

# # def EOD_score(estimator, X, y_true, bias_colname, unprivileged_value, privileged_value):
# #   y_pred = estimator.predict(X)
# #   if len(y_pred) != len(y_true):
# #     raise Exception("not equal length")
# #   bias_sensitive_arr = X[bias_predictor_name]
# #   EOD_abs = compute_abs_equality_of_opportunity_difference(bias_colname, bias_sensitive_arr, unprivileged_value, privileged_value, y_true, y_pred)
# #   return EOD_abs
  
# # def bias_results_per_biased_predictor(estimator, X, y_true, bias_groups):
# #   for bias_group in bias_groups:
# #     bias_colname = 
# #     unprivileged_values = 
# #     privileged_values = 
# #     #sanity check
# #     for priviliged_value in privileged_values:
# #       if privileged_value in unprivileged_value:
# #         raise Exception("privileged_value in unpriviliged value, so overlap ", privileged_value, unprivileged_values)
# #     SPD = SPD_score(estimator, X, y_true, bias_colname, unprivileged_values, privileged_values) 
# #     EOD = EOD_score(estimator, X, y_true, bias_colname, unprivileged_value, privileged_value)
# #     bias_group_result = {"SPD": SPD, "EOD": EOD}
# #   bias_group_results.append(bias_group_result)
# #   tryutn nisd


# #priviprivileged_protected_attribute_values and unprivileged_protected_attribute_values should be a numpy array
# def encode_df_and_privileged(df_typed, protected_attribute_name, privileged_protected_attribute_values, unprivileged_protected_attribute_values):
#     #encode_all_the_categories
#     cat_columns = df_typed.select_dtypes(['category']).columns
#     reverse_map = {}

# def getClassifierName(grid_search):
#     return grid_search.best_estimator_.named_steps['classifier'].__class__.__name__

# def find_best_estimator_params(grid_search):
#     results = grid_search.cv_results_
#     mask = results['mean_test_accuracy_score'] > 0.8
#     max_value = results['mean_test_weighted_average_recall'][mask].max()
#     param_indexes = np.where(results['mean_test_weighted_average_recall'] == max_value)
#     if len(param_indexes) != 1:
#         raise Exception("excepted only one match... " , param_indexes)
#     param_index = param_indexes[0][0]
#     params = results['params'][param_index]
#     return params
#     # best_estimators = []
#     # for grid_search in grid_searches:
#     #   best_estimator = findBestEstimator(grid_search)
#     #   best_estimators.append(best_estimator)

# #all the grid_searches used in experiment 1 and 2
# def createParamGridEx3(grid_searches):
#     param_grid_experiment_bias = []
#     for grid_search in grid_searches:
#     estimator_params = find_best_estimator_params(grid_search)
#     classifier_name = getClassifierName(grid_search)
#     param_grid_from_grid_search = {}

#         #encapsulate estimator params with array to run in gridsearch
#         for param_key in estimator_params:
#             param_grid_from_grid_search[param_key] = [estimator_params[param_key]]

#         #adjust the classifier based on the model
#         if classifier_name == 'RandomForestClassifier':
#             param_grid_from_grid_search['classifier'] = [create_random_forest_model()]
#         elif classifier_name == 'LGBMClassifier':
#             param_grid_from_grid_search ['classifier'] = [create_lightgbm_model()]
#         elif classifier_name == 'CatBoostClassifier':
#             # param_grid_from_grid_search  = adjust_naming_param_grid_for_catboost(param_grid_from_grid_search)
#             param_grid_from_grid_search['classifier'] = [create_catboost_model()]
#         else:
#             raise Exception("unknown model cannot convert to param for gridsearch")
#         param_grid_experiment_bias.append(param_grid_from_grid_search)
#     return param_grid_experiment_bias

# # CANT DO THIS BECAUSE OF DATA LEAKAGE
# #   cat_imputer = createCategoricalImputer()
# #   num_imputer = createNumericalImputer()
  
# #   imputer_transformer = ColumnTransformer(
# #       transformers = [
# #           ("nominal_impute", cat_imputer, get_nominal_cols(df)),
# #           ("ordinal_impute", cat_imputer, get_ordinal_cols(df)),
# #           ("numerical_impute", num_imputer, make_column_selector(dtype_include="number"))
# #       ], verbose_feature_names_out=False
# #   )
# #   imputer_transformer.set_output(transform="pandas")
# #   imputer_transformer.fit_transform()
#   #encode the privileged and unpriviliged attributes value of the protected attribute
#     encoded_privileged_protected_attribute_values = np.zeros(len(privileged_protected_attribute_values), dtype=np.int8)
#     encoded_unprivileged_protected_attribute_values = np.zeros(len(unprivileged_protected_attribute_values), dtype=np.int8)
#     for i, cat in enumerate(X[protected_attribute_name].cat.categories):
#         if cat in privileged_protected_attribute_values:
#             encoded_privileged_protected_attribute_values[(privileged_protected_attribute_values==cat)] = int(i)
#         elif cat in unprivileged_protected_attribute_values:
#             encoded_unprivileged_protected_attribute_values[(unprivileged_protected_attribute_values==cat)] = int(i)
#         else:
#             raise Exception("Not all attribute values are inclued of the privilige class miss: ", cat) 

#     #save the encoding map so that the changes can be reverted later
#     for cat_column in cat_columns:
#         #encode values and save the reversemap
#         cat_index = X[cat_column].cat.categories
#         reverse_map[cat_column] = dict(enumerate(cat_index)) #example {0:'F', 1:'M'}
#         # print("****reverse_map: ", reverse_map)

#     #encode all categories to be compatible with the BinaryLabelDataset of aif360
#     df_typed[cat_columns] = df_typed[cat_columns].apply(lambda x: x.cat.codes)
#     return df_typed, encoded_privileged_protected_attribute_values, encoded_unprivileged_protected_attribute_values, reverse_map

# def rebuild_to_correct_format_df(biased_df, fill_value, reverse_map, nominal_application_features, ordinal_application_features_grouped_order_info):
#     df_bias_corrected, attributes_extra_info = biased_df.convert_to_dataframe()
#     #the missing impute stays the same
#     #revert fill_value
#     for col in df_bias_corrected.columns:
#         df_bias_corrected.loc[df_bias_corrected[col] == fill_value, col] = np.nan


#     # revert categorical encoded columns back to original category
#     for col in reverse_map:
#         mapping = reverse_map[col]
#         #apply_mapping to col
#         df_bias_corrected[col] = df_bias_corrected[col].map(mapping)

#     #change dataframe meta information about dtypes
#     df_bias_corrected_typed = changeColTypesOfDataframe(df_bias_corrected, nominal_application_features, ordinal_application_features_grouped_order_info)
#     df_bias_corrected_typed.set_index(pd.RangeIndex(start=0, stop=df_bias_corrected_typed.shape[0], step=1), inplace=True)
#     return df_bias_corrected_typed

# def results_for_repair_levels(binaryLabelDataset, protected_attribute_name, repair_levels, param_grid, reverse_map, fill_value, random_state):
#     #https://aif360.readthedocs.io/en/latest/modules/sklearn.html
#     #   Currently, while all scikit-learn classes will accept DataFrames as inputs, most classes will return a numpy.ndarray. 
#     # Therefore, many pre- processing steps, when placed before an aif360.sklearn step in a Pipeline, will cause errors.
#     # print(binaryLabelDataset)
#     grid_searches = []
#     #print("encoded_df *****", encoded_df)
#     #each repaired bias set need to be fit through the pipeline separately

#     for repair_level in repair_levels:
#         print("***** starting repair ****")
#         #src: https://towardsdatascience.com/mitigating-bias-in-ai-with-aif360-b4305d1f88a9
#         di = DisparateImpactRemover(repair_level = repair_level, sensitive_attribute=protected_attribute_name)
#         #fit_transform in aif360 will return a new dataset:
#         #src: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.Transformer.html
#         #Return a new dataset generated by running this Transformer on the input.
#         print("transforming...")
#         dataset_transf_train = di.fit_transform(binaryLabelDataset)
#         print("end transform...")
#         # print(dataset_transf_train)
#         # adapted_ordinal_application_features_grouped_order_info = [
#         #   {"columnName": 'NAME_EDUCATION_TYPE', "categories": [0.,1.,2.,3.,4.]},
#         #   {"columnName": 'REGION_RATING_CLIENT_W_CITY', "categories": [1,2,3]}
#         # ]
#         print("rebuilding ...")
#         df_bias_corrected_formatted = rebuild_to_correct_format_df(dataset_transf_train, fill_value, reverse_map, RAW_nominal_application_features, RAW_ordinal_application_features_grouped_order_info)
#         X_bias_corrected_formatted = df_bias_corrected_formatted.drop(columns=['TARGET'])
#         print("end rebuilding...")
#         #print(X_bias_corrected_formatted['CODE_GENDER'])
#         # print(len(X_bias_corrected_formatted))
#         #convert pack to nan
#         if X_bias_corrected_formatted.shape[0] != X.shape[0] and X_bias_corrected_formatted.shape[1] != X.shape[1]:
#             raise Exception("unequal shapes for biased X: bias, x ", X_bias_corrected_formatted.shape, X.shape)
#         if y.shape[0] != X_bias_corrected_formatted.shape[0]:
#             raise Exception("unequal rows for y and X_bias", y.shape, X_bias_corrected_formatted.shape)
#         # print(X_bias_corrected_formatted.shape, y.shape)
#         grid_search = build_grid_search(estimator='passthrough', X=X_bias_corrected_formatted, param_grid=param_grid, random_state=random_state)
#         # print(X_bias_corrected_formatted.shape, y.shape)
#         print("starting the grid_search for repair_level ...: ", repair_level)
#         grid_search.fit(X_bias_corrected_formatted, y)
#         print("****grid search completed with: estimator, params, repair_level", grid_search.best_estimator_.named_steps['classifier'], param_grid, repair_level)
#         grid_searches.append(grid_search)
#         print("--------END REPAIR-----------")
#         #bias_results_per_biased_predictor(estimator, X, y_true, bias_groups)
    
#     return grid_searches


# def loadDatasetIntoAif360Format(encoded_df, protected_attribute_name, privileged_protected_attribute, unprivileged_protected_attribute, fill_value):
#     # print(encoded_df['NAME_EDUCATION_TYPE'])
#     # temporary fill all NaN values

#     for col in encoded_df:
#     if fill_value in encoded_df[col].values:
#       raise Exception("fill value is contained in col: ", col)
    
    
#     encoded_df.fillna(value=fill_value, axis=1,inplace=True)
#     # print(privileged_protected_attribute, "un:", unprivileged_protected_attribute)
#     # print(type(privileged_protected_attribute[0]), "un:", type(unprivileged_protected_attribute[0]))
#     binaryLabelDataset = datasets.BinaryLabelDataset(
#       favorable_label=1,
#       unfavorable_label=0, 
#       df=encoded_df,
#       label_names=['TARGET'], 
#       protected_attribute_names=[protected_attribute_name],
#       privileged_protected_attributes = [privileged_protected_attribute],
#       unprivileged_protected_attributes = [unprivileged_protected_attribute]
#     )
#     return binaryLabelDataset


# def perform_experiment_3(grid_searches, repair_levels, X, y):
#     #both for undersampling and class_weights are done
#     param_grid_experiment_bias = createParamGridEx3(grid_searches)

#     #sanity check
#     if len(param_grid_experiment_bias) != len(grid_searches):
#     raise Exception("something went wrong with teh param_grid_experiment_bias, please check")
#     #prepare the dataset
#     # print(param_grid_experiment_bias)
#     #define sensitive attribute and characteristics

#     #params_gender_scorer is defined in the fair metrics section
#     #for consistency we use the values saved there
#     protected_attribute_name = params_gender_scorer['bias_colname']
#     privileged_protected_attribute_values = np.array(params_gender_scorer['privileged_values']) #must be np.array (format required by aif360)
#     unprivileged_protected_attribute_values = np.array(params_gender_scorer['unprivileged_values']) #must be np.array (format required by aif360)

#     #fill value
#     fill_value = -2000000

#     df_X_y = X.copy()
#     df_X_y['TARGET'] = y
#     encoded_df, encoded_privileged_protected_attribute_values, encoded_unprivileged_protected_attribute_values, reverse_map = encode_df_and_privileged(
#     df_X_y,
#     protected_attribute_name, 
#     privileged_protected_attribute_values, 
#     unprivileged_protected_attribute_values
#     )
#     #load into dataset format of aif360
#     binaryLabelDataset = loadDatasetIntoAif360Format(encoded_df=encoded_df, 
#                                                    protected_attribute_name=protected_attribute_name, 
#                                                    privileged_protected_attribute=encoded_privileged_protected_attribute_values, 
#                                                    unprivileged_protected_attribute=encoded_unprivileged_protected_attribute_values, 
#                                                    fill_value=fill_value)
#     results = results_for_repair_levels(binaryLabelDataset=binaryLabelDataset, protected_attribute_name=protected_attribute_name, repair_levels=repair_levels, param_grid
#                                         =param_grid_experiment_bias, reverse_map=reverse_map, fill_value=fill_value, random_state = RANDOM_SPLIT_STATE)
#     #each is the result of all params being fit on one dataset (each dataset is different per repair level).
#     return results
