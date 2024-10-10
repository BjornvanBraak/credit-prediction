#import metrics

# imputation
from sklearn.impute import SimpleImputer

# encoding and scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder

#over and undersampling
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import NearestNeighbors

#pipeline
from imblearn.pipeline import Pipeline as Imblearn_Pipeline
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def createNumericalImputer():
    return SimpleImputer(strategy='median')

def createCategoricalImputer():
    return SimpleImputer(strategy='most_frequent')

#Multiple vs. Single Imputation
# It is still an open problem as to how useful single vs. multiple imputation is 
# in the context of prediction and classification when the user is not interested in measuring uncertainty due to missing values.

#Optional improvement with multivariate imputation
# Any missing values were imputed using the MissForest algorithm due to its robustness in the face of multicollinearity, 
# outliers, and noise. Categorical predictors were one-hot encoded using pandas get_dummies 
# function with no dropped subtypes (drop_first=False). Low-information variables (e.g., ID numbers, etc.) 
# were dropped prior to the train/test partition.
#https://towardsdatascience.com/the-mystery-of-feature-scaling-is-finally-solved-29a7bb58efc2

#the SMOTE-NC algorithm will encode all categorical values given with onehotencoding
#therefore the nominal values do not need to be encoded here

#the ordinal values REGION_RATING_CLIENT and REGION_RATING_CLIENT_W_CITY are already numerical in range 1,2,3

# NEW IN VERSION 3
def check_allowed_json_character_in_col_names_lightgbm(X_columns):
  #for more info: https://github.com/microsoft/LightGBM/blob/18dbd65e57995618ee2a8b1f7e4cb0df1f9c6333/include/LightGBM/utils/common.h#L886-L902
       # original condition in lightgbm code:
    #      if (char_code == 34      // "
    #     || char_code == 44   // ,
    #     || char_code == 58   // :
    #     || char_code == 91   // [
    #     || char_code == 93   // ]
    #     || char_code == 123  // {
    #     || char_code == 125  // }
    #     ) {
    #   return false;
    # }
    valid = True
    not_allowed_dict = {}
    for col in X_columns:
        for char in col:
            char_code = ord(char)
            if (char_code == 34    
              or char_code == 44
              or char_code == 58
              or char_code == 91
              or char_code == 93
              or char_code == 123
              or char_code == 125):
                not_allowed_dict[col] = char
                valid = False

 
        
    return valid, not_allowed_dict

def feature_name_combiner(input_feature, category):
    return re.sub(r'"|,|:|[|]|{|}', '/', input_feature)


#the ordinal feature NAME_EDUCATION_TYPE is not encoded the OrdinalEncoder will be used to transform this data
def createOrdinalEncoder(categories):
    # print("ordinalencoder cat ", categories)
    #handle_unknown=
    #should still be handled correclty when basic model is debugged
    #not sure if the handling of unknown values is correct
    return OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value', unknown_value=-1)

def createNumericLabeler():
    return OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int32)


def createOneHotEncoder():
    #possible memory optimisation with converting to dtype=np.float32
    #dtype=
    #since 
    #for DT: The input samples. Internally, it will be converted to dtype=np.float32 and if a sparse matrix is provided to a sparse csr_matrix.
    #????????????????
    #handle_unknown=
    #should sitll be handled correctly when basic model is debugged

    #https://towardsdatascience.com/dealing-with-features-that-have-high-cardinality-1c9212d7ff1b
    #high cardinality problem

    #onehotencoding is also not good for the use in decision tree based models
    # NEW IN V3
    #sparse_output: reason pd dataframe cannot be sparse in with sklearn transform_ouput='pandas'
    #feature_name_combiner: reason lightgbm does not except certain feature names, see check_allowed_json_character_in_col_names_lightgbm for more info
    return OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False, feature_name_combiner=feature_name_combiner)


def createScaler():
    return MinMaxScaler(copy=False)

def createSmote(categorical_features, random_state, categorical_encoder):
    #concatenates the strings (makes a copy)
    #N_min/N_maj = 0.25 --> brings to 20% N_min of the total_sample (~200% oversampling; original was 8.9%)
    #NOTE NO REAL JUSTIFICATION GIVEN OTHER THAN THE ORIGINAL PAPER ALSO OVERSAMPLING IN THIS RANGE

    #NOTE NEED TO SET THE n_jobs of the k_neighbors since -->
    #Deprecated since version 0.10: n_jobs has been deprecated in 0.10 and
    #will be removed in 0.12. It was previously used to set n_jobs of nearest neighbors algorithm. 
    #From now on, you can pass an estimator where n_jobs is already set instead.
    k_neighbors = NearestNeighbors(n_jobs=-1)

    #NOTE OTHER POSSIBLE SPEEDUPS:
    # use cuml
    # https://medium.com/rapids-ai/faster-resampling-with-imbalanced-learn-and-cuml-6cfc1dae63bf
    #sample_strategy is chosen during the grid_search
    return SMOTENC(categorical_features=categorical_features, k_neighbors=k_neighbors, random_state=random_state, categorical_encoder=categorical_encoder)

def createRandomUnderSampling(random_state):
    #N_min/N_maj = 0.5 --> brings to 33% N_min of the total_sample
    #NOTE NO REAL JUSTIFICATION GIVEN FOR THE PROPORTION CHOSEN
    return RandomUnderSampler(random_state=random_state)

from sklearn.preprocessing import PolynomialFeatures
def createPolynomialFeatures():
    return PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

#PIPELINE
# DEPRECATED IN V3
# def get_nominal_cols(df):
#     categorical_cols = df.select_dtypes(include='category').columns
#     indexes = []
#     for ordinal_feature in ordinal_application_features:
#         idx = categorical_cols.get_loc(ordinal_feature)
#         indexes.append(idx)
#     nominal_cols = categorical_cols.delete(indexes)
#     return nominal_cols

# def get_ordinal_cols(df):
#     for feature in ordinal_application_features:
#         if not (feature in df.columns):
#             raise Exception("feature is not in col of df ", feature)
#     return ordinal_application_features


def build_preprocessing_steps(nominal_features, ordinal_features_dict, discrete_features, continuous_features):
        #The focus in this function is on essembling the steps together into a pipeline
    #The specific decision made in the 'Data preprocessing' section are not discussed.
    #The general pipeline steps:
    # - impute missing values
    # - encode
    #https://towardsdatascience.com/pipeline-columntransformer-and-featureunion-explained-f5491f815f
    #new plan make two pipelines one for num the other for cat
    #also create the transformers here also gives more clarity
    #instead of in the data preprocessing also gives a lot more clarity
    #it abstract away the exact thing done in each step
    #however, it does keep the columntransformer and the pipeline
    #in here so you have a clue what is going on
    #then try and figure out how to get the categorical colunmn places

    #base number of cols is:
    # DEPRECATED IN V3
    # num_cols = X.shape[1]
    numerical_pipeline = Pipeline(steps=[('impute', createNumericalImputer()), ('scale', createScaler())])

    #ORDINAL PIPELINE

    # DEPRECATED IN V3
    # ordinal_feature_names = get_ordinal_cols(df=X)
      
    # ordered_categories_pre_transform = findCategories(df=X, colNames=ordinal_feature_names)
    # if len(ordered_categories_pre_transform) != len(ordinal_feature_names):
    #     raise Exception("different lengths for ordinal features:names, categories ", len(ordinal_feature_names), len(ordered_categories_pre_transform))
    #     # print("ordered_categories_pre_transform ", ordered_categories_pre_transform)
    
    categories = list(ordinal_features_dict.values())
    ordinal_features = list(ordinal_features_dict.keys())
    ordinal_pipeline = Pipeline(steps=[('impute', createCategoricalImputer()), ('encode', createOrdinalEncoder(categories=categories)), ('scale', createScaler())])

    #NOMINAL PIPELINE
    #the nominal feature are encoded by the smote algorithm, they only need to be imputed
    
    # DEPRECATED IN V3
    # print("**** NOMINAL INDEXES ****", nominal_indexes_pre_transform)
    # nominal_feature_names = get_nominal_cols(df=X)
    
    nominal_pipeline = Pipeline(steps=[('impute', createCategoricalImputer())])

    
    # DEPRECATED IN V3
    # for the dtypes
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html#pandas.DataFrame.select_dtypes
    # https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html#sklearn.compose.make_column_selector
    # https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
    # if len(ordinal_feature_names) + len(nominal_feature_names) + len(X.select_dtypes(include="number").columns) != len(X.columns):
    #     raise Exception("error occured with the feature names")
    # if len(ordinal_feature_names) != 2:
    #     raise Exception("error occured with the ordinal features: ", ordinal_feature_names)

    #TRANSFORMER FOR PREPROCESSING (ex. SAMPLING)
    #The way a transformer transforms the data is always 
    data_cleaning_and_preparation_transformer = ColumnTransformer(
      transformers = [
          ("nominal_pipe", nominal_pipeline, nominal_features),
          ("ordinal_pipe", ordinal_pipeline, ordinal_features),
          ("numerical_pipe", numerical_pipeline, [*discrete_features, *continuous_features])
      ], verbose_feature_names_out=False
    )
    #the transformer returns an array so no longer can the names or dtypes be used for the columns

    #RANDOM_STATE
    rng_smote = 100
    rng_under = 101

    smote_oversampling = createSmote(categorical_features = nominal_features, random_state=rng_smote, categorical_encoder=createOneHotEncoder())
    random_undersampling = createRandomUnderSampling(random_state=rng_under)

    cat_encoding_and_feature_engineering_transformer = ColumnTransformer(
      transformers = [
          ("nominal_encoding", createOneHotEncoder(), nominal_features),
          ("polynomial_features_num", createPolynomialFeatures(), [*discrete_features, *continuous_features])
      ], remainder='passthrough'
    )
    #DIMENSIONS HAVE BEEN CHANGED BY POLYNOMIAL_FEATURE TRANSFORM
    #HOWEVER THE NOMINAL_INDEXES SHOULD STILL BE FIRSt

    preprocessing_steps = [
    ('data_cleaning_and_preparation', data_cleaning_and_preparation_transformer),
    ('oversample', smote_oversampling),
    ('undersample', random_undersampling),
    ('cat_encoding_and_feature_engineering', cat_encoding_and_feature_engineering_transformer),
    #however the different package have different requirements for what form the data needs to be in
    ] 
    
    return preprocessing_steps

def build_pipeline(estimator, nominal_features, ordinal_features_dict, discrete_features, continuous_features, transform_output='pandas'):
    if transform_output != 'pandas':
        raise Exception('Other outputs than "pandas" are no longer supported in V3')    

    preprocessing_steps = build_preprocessing_steps(nominal_features, ordinal_features_dict, discrete_features, continuous_features)
    
    # https://stackoverflow.com/questions/48370150/how-to-implement-smote-in-cross-validation-and-gridsearchcv?rq=4
    # thePurplePython Yes. You are correct. The imblearn pipeline will only call sample() method on training data and not on test data. The test data will be passed through without any changes. – 
    # Vivek Kumar
    #  May 21, 2019 at 5:48
    # --> The imblearn pipeline is responsible for invoking the fit_resample method only on the train datase
    
    # imblearn_pipeline = Imblearn_Pipeline(steps=all_steps)
    imblearn_pipeline = Imblearn_Pipeline(steps=[*preprocessing_steps, ('classifier', estimator)])
    # print("********pipeline_build...")
    imblearn_pipeline.set_output(transform=transform_output)
    
    return imblearn_pipeline

def calculate_total_jobs(param_grid, n_repeats, n_splits):
    if isinstance(param_grid, dict):
        return len(param_grid)
    elif isinstance(param_grid, list):
        total = 1
        for dictionairy in param_grid:
            if len(dictionairy) > 1:
                total *= len(dictionairy)
            else:
                raise Exception('unexpected dictionairy length, could not calculate the total jobs spawmned')
        #each unique gridsearch pair * cv_runs
        total *= n_repeats * n_splits
        return total
    
import joblib
import psutil
import math
import pprint

# these parrelellisaiton will be beneficially on larger datasets.
# and can be used independently of the classifier used in the process.
def calculate_optimal_n_jobs_and_pre_dispatch(param_grid, n_repeats, n_splits, dataset_size_GB):
    total_jobs = calculate_total_jobs(param_grid, n_repeats, n_splits)
    print(f'total jobs for gridsearch are: {total_jobs}')
    
    total_cpu_count = joblib.cpu_count()
    available_virtual_memory_in_GB = psutil.virtual_memory().available >> 30
    
    if available_virtual_memory_in_GB <= 2:
        raise Exception('extremely low virtual memory available: ' + available_virtual_memory_in_GB)
    
    max_n_jobs_based_on_memory = math.floor(available_virtual_memory_in_GB / dataset_size_GB)
    
    pprint.pprint({
        "availabe_cpu": total_cpu_count,
        "available_virtual_memory": available_virtual_memory_in_GB,
        "max_n_jobs_memory_bound": max_n_jobs_based_on_memory,
        "max_n_jobs_cpu_bound": total_cpu_count
    })
    
    # the maximum number of jobs possible to spamn
    if max_n_jobs_based_on_memory > 2 * total_cpu_count:
        #the most optimal setting
        n_jobs = math.ceil(total_cpu_count-0.05*total_cpu_count)
        pre_dispatch = '2*n_jobs'
    elif max_n_jobs_based_on_memory <= 2 * total_cpu_count  and max_n_jobs_based_on_memory >= total_cpu_count:
        n_jobs = math.ceil(total_cpu_count-0.05*total_cpu_count)
        value = round(max_n_jobs_based_on_memory / total_cpu_count,4)
        pre_dispatch = f'{str(value)}*n_jobs'
    else: #max_n_jobs_based_on_memory < total_cpu_count
        n_jobs = max_n_jobs_based_on_memory
        pre_dispatch = max_n_jobs_based_on_memory
    
    if n_jobs > total_jobs:
        n_jobs = total_jobs
    
    return n_jobs, pre_dispatch
    
    

#define gridsearch
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
def build_grid_search(estimator, param_grid, random_state, evaluation_metrics, nominal_features, ordinal_features_dict, discrete_features, continuous_features, set_optimal_n_jobs=True, n_jobs=None, verbose=0, dataset_size_GB=None):
    if set_optimal_n_jobs and n_jobs != None:
        raise Exception('you cannot set_optimal_n_jobs')
        
    if set_optimal_n_jobs and not dataset_size_GB:
        raise Exception('in order to calculate the optimal_n_jobs the dataset_size_GB should be given')
    
    if set_optimal_n_jobs:
        print('Warning: in case of set_optimal_n_jobs=True, the assumptions is that estimators used in grid_search have n_jobs ~= 1')
    
    #Shared across model tuning:
    #metrics used in scoring
    #cross validation procedure
    n_repeats=5
    n_splits=2
    
    repeated_stratified_kfold = RepeatedStratifiedKFold(
        n_repeats=n_repeats,
        n_splits=n_splits,
        random_state=random_state
    ) #cv
    
    
    #build pipeline
    pipeline = build_pipeline(
        estimator = estimator, 
        nominal_features=nominal_features, 
        ordinal_features_dict=ordinal_features_dict, 
        discrete_features=discrete_features, 
        continuous_features=continuous_features
    )

    # NOTES ABOUT MEMORY CONSUMPTION
    #   memory consumption gridsearch:
    #   Suppose you are using GridSearchCV for KNN with parameters' grid: k=[1,2,3,4,5, ... 1000].
    # Even when you set n_jobs=2, GridSearchCV will first create 1000 jobs, each with one choice of your k, also making 1000 copies of your data (possibly blowing up your memory if your data is big), then sending those 1000 jobs to 2 CPUs (most jobs will be pending of course).
    # GridSearchCV doesn't just spawn 2 jobs for 2 CPUs because the process of spawing jobs on-demand is expensive. It directly spawns equal amount of jobs as parameter combinations you have (1000 in this case).
    # In this sense, the wording n_jobs might be misleading. Now, using pre_dispatch you can set how many pre-dispatched jobs you want to spawn.

    # n_jobs = -1
    # pre_dispatch = 2*n_jobs (default)
    # blowing up memory in the case of 25
    # since there are 144 cpus available
    # therefore -->
    # 144 * 2 = 288 *
    
    n_jobs=n_jobs,
    pre_dispatch='2*n_jobs', #is the default
    
    if set_optimal_n_jobs:
        n_jobs, pre_dispatch =calculate_optimal_n_jobs_and_pre_dispatch(param_grid, n_repeats, n_splits, dataset_size_GB)
        
    pprint.pprint({
        'n_jobs': n_jobs,
        'pre_dispatch': pre_dispatch
    })
    
    
    if n_jobs > joblib.cpu_count():
        raise Exception(f'too high n_jobs, available cpu are: {joblib.cpu_count()}')
    
    
    grid = GridSearchCV(estimator=pipeline,
                 param_grid=param_grid,
                 scoring=evaluation_metrics, 
                 refit="roc_auc",
                 cv=repeated_stratified_kfold, 
                 error_score='raise',
                 n_jobs=n_jobs,
                 pre_dispatch=pre_dispatch,
                 verbose=verbose
                 )
    return grid