
#define evaluation metrics
from sklearn.metrics import make_scorer, roc_curve, roc_auc_score, fbeta_score, accuracy_score, cohen_kappa_score, balanced_accuracy_score

#the following naming convention was followed for naming scorer functions (based on sklearn)
#The module sklearn.metrics also exposes a set of simple functions measuring a prediction error given ground truth and prediction:
#functions ending with _score return a value to maximize, the higher the better.
#functions ending with _error or _loss return a value to minimize, the lower the better. When converting into a scorer object using make_scorer, set the greater_is_better parameter to False (True by default; see the parameter description below).



# - performance metrics
from sklearn.metrics import recall_score


#https://arxiv-org.ezproxy2.utwente.nl/pdf/2010.05995.pdf
def compute_weighted_average_recall_score(y_true, y_pred):
  #order of recall_socres will be determined by labels, so first bad lenders
  #then the good lenders recall score will be returned
  recall_scores = recall_score(y_true, y_pred, average=None, labels=[0, 1])
  tpr = recall_scores[1]
  tnr = recall_scores[0]

  freq_good = len(y_true[y_true == 1])
  freq_bad = len(y_true[y_true == 0]) 
  freq_total = freq_bad + freq_good

  #what to do when the thing is zero? 
  sum_freq = (1/freq_good)+(1/freq_bad)
  w_good = 1/(freq_good*sum_freq)
  w_bad = 1/(freq_bad*sum_freq)
  
  weighted_average_class_accuracy = (w_good*tpr+w_bad*tnr)
  # print("weight good, tpr, weight bad, tnr", w_good, tpr, w_bad, tnr)
  # print("total weighted average class accuracy", weighted_average_class_accuracy)
  return weighted_average_class_accuracy

#create f-measure
beta = 1
fmeasure_scorer = make_scorer(fbeta_score, beta=beta)

#create balanced accuracy
weighted_average_recall_scorer = make_scorer(compute_weighted_average_recall_score)

#roc_auc_score is complete
roc_auc_scorer = make_scorer(roc_auc_score)

#accuracy
accuracy_scorer = make_scorer(accuracy_score)

cohen_kappa_scorer = make_scorer(cohen_kappa_score)

balanced_accuracy_sklearn_scorer = make_scorer(balanced_accuracy_score)


# - fairness metrics

from functools import partial

# def calculatePofAGivenBC(df, A_colname, B_colname, C_colname, A_value, B_value, C_values):
#   # print("calculatingPOfAGivenBC...")
#   num_A_B_C = df[(df[A_colname]==(A_value))&(df[B_colname] == B_value)&(df[C_colname].isin(C_values))].shape[0]
#   num_B_C = df[(df[B_colname]==(B_value))&(df[C_colname].isin(C_values))].shape[0]
#   # print("num_ABC", num_A_B_C)
#   # print("num_B_C: ", num_B_C)
#   return num_A_B_C / num_B_C
# #am I going to do it on the best_estimators?
# #or on all estimators
# #definetely only on the best estimators
# #I can do it afterwards by predicting with the best model estimators
# #and applying something with and without doing something to the dataset?

# #score function for EOD
# def compute_abs_equality_of_opportunity_difference(y_true, y_pred, bias_colname, bias_sensitive_arr, unprivileged_values, privileged_values):
#   if len(y_true) != len(y_pred):
#     raise Exception("unequal length :" , y_true, y_pred)
#   df = pd.DataFrame({bias_colname:bias_sensitive_arr, 'y_true':y_true, 'y_pred':y_pred})
#   #https://stats.stackexchange.com/questions/67318/definition-of-conditional-probability-with-multiple-conditions
#   p_yhat1_y1_Aunpriveliged = calculatePofAGivenBC(df, 'y_pred', 'y_true', bias_colname, 1, 1, unprivileged_values)
#   p_yhat1_y1_Apriveliged = calculatePofAGivenBC(df, 'y_pred', 'y_true', bias_colname, 1, 1, privileged_values)
#   EOD = p_yhat1_y1_Aunpriveliged - p_yhat1_y1_Apriveliged 
#   # print("EOD: " , EOD)
#   return abs(EOD)

# def calculatePofAGivenB(df, A_colname, B_colname, A_value, B_values):
#   # print("calculatingPOfAGivenB...")
#   # print(df)
#   # print("colnameA:", A_colname, "colnameB: ", B_colname, "A_value: ", A_value, "B_value: ", B_values)
#   num_A_B = df[(df[A_colname]==A_value)&(df[B_colname].isin(B_values))].shape[0]
#   num_B = df[(df[B_colname].isin(B_values))].shape[0]
#   # print("num_AB", num_A_B)
#   # print("num_B: ", num_B)
#   return num_A_B / num_B

# #score function for SPD
# def compute_abs_statistical_parity_difference(y_true, y_pred, bias_colname, bias_sensitive_arr, unprivileged_values, privileged_values):
#   if len(y_true) != len(y_pred):
#     raise Exception("unequal length :" , y_true, y_pred)
#   y_true = "should be empty"
#   df = pd.DataFrame({bias_colname:bias_sensitive_arr, 'y_pred':y_pred})
#   #https://stats.stackexchange.com/questions/67318/definition-of-conditional-probability-with-multiple-conditions
#   p_yhat1_Aunpriveliged = calculatePofAGivenB(df=df, A_colname='y_pred', B_colname=bias_colname, A_value=1, B_values=unprivileged_values)
#   p_yhat1_Apriveliged = calculatePofAGivenB(df=df, A_colname='y_pred', B_colname=bias_colname, A_value=1, B_values=privileged_values)
#   SPD = p_yhat1_Aunpriveliged - p_yhat1_Apriveliged 
#   # print("SPD: ", SPD)
#   return abs(SPD)


# #base of custom_bias_scorer
# def custom_bias_scorer(estimator, X, y, **kwargs):
#   #https://github.com/scikit-learn/scikit-learn/blob/364c77e047ca08a95862becf40a04fe9d4cd2c98/sklearn/metrics/_scorer.py#L247
#   #inspired by
#   #kwargs are decomposed for clarity
#   y_true = y
#   y_pred = estimator.predict(X)
#   _score_func = kwargs['compute_bias_score']
#   bias_colname = kwargs['bias_colname']
#   bias_sensitive_arr = X[bias_colname]
#   unprivileged_values = kwargs['unprivileged_values']
#   privileged_values = kwargs['privileged_values']
#   greater_is_better = kwargs['greater_is_better']
#   sign = 1 if greater_is_better else -1
  
#   # print(bias_sensitive_arr)
#   # print("******shape of X in scorer: ", X.shape)
#   #calculate the score
#   # the sklearn learn convention is followed that all score will be 
#   # When converting into a scorer object using make_scorer, set the greater_is_better parameter to False (True by default; see the parameter description below).
#   # the underneath code performs a sign flip, this is needed to 
#   # this is needed in order for rank_test_score_x to work properly
#   return sign * _score_func(y_true, y_pred, bias_colname, bias_sensitive_arr, unprivileged_values, privileged_values)

# #bind paramaters related to bias predictor to scorer function
# def create_custom_bias_scorer(compute_bias_score, bias_colname, unprivileged_values, privileged_values, greater_is_better):
#   #bind paramaters to function
#   #the generic scorer function is now converted to a scorer assigned to biased predictor
#   bound_params_custom_scorer = partial(custom_bias_scorer,
#                                        compute_bias_score = compute_bias_score,
#                                        bias_colname = bias_colname,
#                                        unprivileged_values = unprivileged_values,
#                                        privileged_values = privileged_values,
#                                        greater_is_better = greater_is_better)
#   return bound_params_custom_scorer

# #create the fairness scorer for a specific biased predictor
# def createFairnessScorersPerBaisedPredictor(bias_colname, unprivileged_values, privileged_values):
#   #following the sklearn naming convention
#   neg_abs_spd_loss_scorer = create_custom_bias_scorer(compute_abs_statistical_parity_difference, bias_colname, unprivileged_values, privileged_values, False)
#   neg_abs_eod_loss_scorer = create_custom_bias_scorer(compute_abs_equality_of_opportunity_difference, bias_colname, unprivileged_values, privileged_values, False)
#   return neg_abs_spd_loss_scorer, neg_abs_eod_loss_scorer

# #define all evaluation metrics
# from sklearn.metrics import make_scorer, roc_curve, roc_auc_score, fbeta_score, accuracy_score, cohen_kappa_score, balanced_accuracy_score

# #the following naming convention was followed for naming scorer functions (based on sklearn)
# #The module sklearn.metrics also exposes a set of simple functions measuring a prediction error given ground truth and prediction:
# #functions ending with _score return a value to maximize, the higher the better.
# #functions ending with _error or _loss return a value to minimize, the lower the better. When converting into a scorer object using make_scorer, set the greater_is_better parameter to False (True by default; see the parameter description below).


# #the values encode pre pipeline
# params_gender_scorer = {
#   "bias_colname":'CODE_GENDER', 
#   "unprivileged_values":['F', 'XNA'], 
#   "privileged_values":['M']
# }
# neg_abs_spd_loss_GENDER_scorer, neg_abs_eod_loss_GENDER_scorer = createFairnessScorersPerBaisedPredictor(**params_gender_scorer)

# params_education_scorer = {
#   "bias_colname":'NAME_EDUCATION_TYPE', 
#   "unprivileged_values":['Lower secondary', 'Secondary / secondary special', 'Incomplete higher'], 
#   "privileged_values": ['Higher education', 'Academic degree']
# }
# neg_abs_spd_loss_EDUCATION_scorer, neg_abs_eod_loss_EDUCATION_scorer = createFairnessScorersPerBaisedPredictor(**params_education_scorer)

# params_region_scorer = {
#   "bias_colname":'REGION_RATING_CLIENT_W_CITY', 
#   "unprivileged_values":[1,2], 
#   "privileged_values":[3]
# }
# neg_abs_spd_loss_REGION_scorer, neg_abs_eod_loss_REGION_scorer = createFairnessScorersPerBaisedPredictor(**params_region_scorer)



# #fairness
# #the fairness metric(s) will not be computed through sklearn scorer object. The reason is that sklearn either expects a value that should be minimized (--> 0)
# #or maximized (--> 1). In the fairness metric case it should approach a value, for example in the case of EOD (--> 0 <--), the closer to that value the better.
# #to prevent misinterpreting and for example calling the best_score_ 
# #sklearn docs:
# #https://scikit-learn.org/stable/modules/model_evaluation.html
# #For the most common use cases, you can designate a scorer object with the scoring parameter; 
# #the table below shows all possible values. All scorer objects follow the convention that higher return values are better 
# #than lower return values. Thus metrics which measure the distance between the model and the data, like metrics.mean_squared_error, 
# #are available as neg_mean_squared_error which return the negated value of the metric.