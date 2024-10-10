from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


#helpers:

def shortDescriptionSampleStrategy(sample_strategy, oversample_strategies_map):
    descr_param = None
    for known_strategy, description in sample_strategy_map.items():
        if math.isclose(sample_strategy, known_strategy):
            return str(description)
            
    #if not returned raise exception
    raise Exception("unknown sample strategy ", sample_strategy)

def shortDescriptionClassWeight(class_weight, class_weights_map):
    descr_param = None
    if str(class_weight) in class_weights_map:
        descr_param = class_weights_map[str(class_weight)]
    else:
        raise Exception("unknown weight applied in map, class_weight", class_weights_map, class_weight)
    return descr_param

def shortDescriptionUndersamplingRatio(undersampling_ratio, undersampling_ratios_map):
    descr_param = None
    if undersampling_ratio in undersampling_ratios_map:
        descr_param = undersampling_ratios_map[undersampling_ratio]
    else:
        raise Exception("unknown undersampling ratio applied: ", undersampling_ratio)
    return descr_param

def getOverSampleStrategy(param):
    if 'oversample' in param and param['oversample'] == 'passthrough':
        return 0
    elif 'oversample__sampling_strategy' in param:
        return param['oversample__sampling_strategy']
    else:
        raise Exception("Unknown sample strategy in param: ", param)

    
def getClassWeight(param):
    if 'classifier__class_weight' in param:
        return param['classifier__class_weight']
    elif 'classifier__class_weights' in param: #catboost has a different name for class_weights
        return param['classifier__class_weights']
    else:
        raise Exception("COULD NOT ACCESS THE CLASS WEIGHT IN PARAM: ", param)
    

def getUndersampleRatio(param):
    if 'undersample__sampling_strategy' in param:
        return param['undersample__sampling_strategy']
    else:
        raise Exception("COULD NOT ACCESS THE UNDERSAMPLING STRATEGY IN PARAM: ", param)
    return
  
def shortParamsExperimentClassWeight(param, oversample_strategies_map, class_weights_map):
    #get specific config
    sample_strategy = getOverSampleStrategy(param)
    class_weight = getClassWeight(param)
    #translate config to description
    descr = ""
    descr += str(shortDescriptionSampleStrategy(sample_strategy, sample_stategies_map))
    descr += "__"
    descr += shortDescriptionClassWeight(class_weight, class_weight_map)
    return descr

def shortParamsExperimentUndersampling(param, oversample_strategies_map, undersampling_ratios_map):
    sample_strategy = getOverSampleStrategy(param)
    undersampling_ratio = getUndersampleRatio(param)
    #translate config to description
    descr = ""
    descr += str(shortDescriptionSampleStrategy(sample_strategy, oversample_strategies_map))
    descr += "__"
    descr += str(shortDescriptionUndersamplingRatio(undersampling_ratio, undersampling_ratios_map))
    return descr

# def shortParamsDescription(experiment_name, params):
#     descr_params = []
#     for param in params:
#         descr_param = ""
#         if experiment_name == "CLASS_WEIGHT":
#             descr_param = shortParamsExperimentClassWeight(param)
#         elif experiment_name == "UNDERSAMPLE":
#             descr_param = shortParamsExperimentUndersampling(param)
#         else:
#             raise Exception("unknown experiment ", experiment_name)
#         descr_params.append(descr_param)
#     return descr_params

def translateToReadableParamsNamesArrays(experiment_name, params, oversample_strategies_map, experiment_ratio_to_name_map):
    sample_arr = []
    experiment_arr = []
    for param in params:
        sample_strategy = getOverSampleStrategy(param)
        #sample_param = shortDescriptionSampleStrategy(sample_strategy)
        sample_arr.append(sample_strategy)
        experiment_param = None
        if experiment_name == "CLASS_WEIGHT":
            class_weight = getClassWeight(param)
            experiment_param = shortDescriptionClassWeight(class_weight, experiment_ratio_to_name_map)
        elif experiment_name == "UNDERSAMPLE":
            undersampling_ratio = getUndersampleRatio(param)
            experiment_param = shortDescriptionUndersamplingRatio(undersampling_ratio, experiment_ratio_to_name_map)
        else:
            raise Exception("unknown experiment ", experiment_name)
        experiment_arr.append(experiment_param)
    return sample_arr, experiment_arr

def translateParamsToDf(experiment_name, params, oversample_strategies_map, experiment_ratio_to_name_map):
    sample_arr, experiment_arr = translateToReadableParamsNamesArrays(experiment_name, params, oversample_strategies_map, experiment_ratio_to_name_map)
    return pd.DataFrame(list(zip(sample_arr, experiment_arr)), columns=["oversample", experiment_name.lower()])

def translateMeanMetricResults(grid_search, metrics):
    all_mean_results = []
    for metric in metrics:
        mean_result_name = 'mean_test_' + metric
        results = grid_search.cv_results_[mean_result_name]
        all_mean_results.append(results)
    transposed = np.array(all_mean_results).T
    return pd.DataFrame(transposed, columns=metrics)

def grid_result_summary_df(experiment, grid_search, metrics, oversample_strategies_map, experiment_ratio_to_name_map):
    grid_results_df = pd.concat([translateParamsToDf(experiment, grid_search.cv_results_["params"], oversample_strategies_map, experiment_ratio_to_name_map),
                          translateMeanMetricResults(grid_search, metrics)],
                          axis=1)
    return grid_results_df


def encode_grid_summary(experiment_name, grid_summary):
    grid_summary_copy = grid_summary.copy()
    encoded_values = []
    if experiment_name == 'CLASS_WEIGHT':
        for class_weight in grid_summary_copy['class_weight']:
            encoded_value = None
            if class_weight == "none":
                encoded_value = 0
            elif class_weight == "little":
                encoded_value = 0.25
            elif class_weight == "moderate":
                encoded_value = 0.5
            elif class_weight == "high":
                encoded_value = 0.75
            elif class_weight == "balanced":
                encoded_value = 1
            else:
                raise Exception("unknown weight applied ", class_weight)
            encoded_values.append(encoded_value)
    elif experiment_name == 'UNDERSAMPLE':
        #already encoded do nothing
        do_nothing = True
    else:
        raise Exception("unknown experiment, ", experiment_name)
    
    grid_summary_copy['class_weight'] = pd.Series(data=encoded_values)
    return grid_summary_copy


beta = 1 #just quick fix
def adjustMetricNameToPrint(metric):
    metricRead = metric
    if metric.lower() == 'accuracy_score':
        metricRead = 'accuracy'
    elif metric.lower() == 'weighted_average_recall':
        metricRead = 'weighted recall'
    elif metric.lower() == 'f_score':
        if beta != None:
            metricRead = 'F'+str(beta)+'-score'
        else:
            raise Exception("beta not defined previously")
    elif metric.lower() == 'roc_auc':
        metricRead = 'ROC AUC'
    else:
        print("unfamiliar metric, metricRead is set to metric")
    return metricRead

def adjustLabelNameToPrint(label):
    labelRead = label
    if label.lower() == 'oversample':
        labelRead = r'$\alpha_{rs}$ over' 
    elif label.lower() == 'undersample':
        labelRead = r'$\alpha_{rs}$ under'
    elif label.lower() == 'class_weight':
        labelRead = label.replace("_", " ")
    else:
        print("unfamiliar label, labelRead is set to label")
    return labelRead


my_gradient_2 = LinearSegmentedColormap.from_list('my_gradient_2', (
    # Edit this gradient at https://eltos.github.io/gradient/#0638F8-4C71FF-FFFFFF-FC4A53-FF000C
    (0.000, (0.024, 0.220, 0.973)),
    (0.250, (0.298, 0.443, 1.000)),
    (0.500, (1.000, 1.000, 1.000)),
    (0.750, (0.988, 0.290, 0.325)),
    (1.000, (1.000, 0.000, 0.047))))

import json


def plot_3d_metric(model_name, grid_summary_encoded, x_label, y_label, metric, all_class_weights, vmin, vmax, ax):
  # 2D-arrays from DataFrame
    df_x = grid_summary_encoded[x_label]
    df_y = grid_summary_encoded[y_label]
    df_z = grid_summary_encoded[metric]
    x1 = np.linspace(df_x.min(), df_x.max(), len(df_x.unique()))
    y1 = np.linspace(df_y.min(), df_y.max(), len(df_y.unique()))
    x2, y2 = np.meshgrid(x1,y1)
    # Interpolate unstructured D-dimensional data.
    z2 = griddata((df_x, df_y), df_z, (x2, y2), method='cubic')
    surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap=my_gradient_2,
                         linewidth=0, antialiased=True, vmin=vmin, vmax=vmax)
    #ax.scatter3D(df_x, df_y, df_z)
    # ax_3d.contour3D(x, y, z, 50, cmap='binary')
    # surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm)

    #adjusting values for reading the plot
    metricRead = adjustMetricNameToPrint(metric)
    x_labelRead = adjustLabelNameToPrint(x_label)
    y_labelRead = adjustLabelNameToPrint(y_label)

    ax.view_init(20, -85)
    ax.set_title(model_name + " - " + metricRead, y=0.93) #y added for visualisation
    ax.set_xlabel(x_labelRead)
    ax.set_ylabel(y_labelRead)
    ax.set_zlim(0, 1)
  
  #the values for class weights are encoded so they can be plotted.
  #However, this does not tell what it entails.
  #the xtickslabels need to be updated
    if x_label == 'class_weight' or y_label == 'class_weight':
        ticks = np.linspace(0, 1, 5)
        labels = []
        for class_weight_dict in all_class_weights:
            if type(class_weight_dict) == str:
                class_weight_dict = json.load(class_weight_dict)
            rounded_dict = {}
            for key in class_weight_dict:
                rounded_dict[key] = round(class_weight_dict[key], 1)
            labels.append(str(rounded_dict))
            ax.set_xlabel(x_labelRead, labelpad=12)

        if x_label == 'class_weight':
            ax.set_xticks(ticks=ticks, labels=labels, rotation=20, fontsize=6)
        elif y_label == 'class_weight':
            ax.set_yticks(ticks=ticks, labels=labels, rotation=20, fontsize=6)
    return ax, surf


def generate_3d_plot(experiment, model_name, grid_search, metric, metric_ranges_map, oversample_strategies_map, experiment_ratio_to_name_map, ax, all_class_weights=None):
    if experiment == "CLASS_WEIGHT":
        x_label = "class_weight"
        y_label = "oversample"
    elif experiment == "UNDERSAMPLE":
        x_label = "oversample"
        y_label = "undersample"
    else:
        raise Exception("unknown experiment ")
    metric_range = metric_ranges_map[metric]
    vmin = metric_range[0]
    vmax = metric_range[1]
    grid_summary = grid_result_summary_df(experiment=experiment, grid_search=grid_search, metrics=[metric], oversample_strategies_map=oversample_strategies_map, experiment_ratio_to_name_map=experiment_ratio_to_name_map)
    grid_summary_encoded = encode_grid_summary(experiment_name=experiment, grid_summary=grid_summary)
    ax, surf = plot_3d_metric(model_name=model_name, grid_summary_encoded=grid_summary_encoded, x_label=x_label, y_label=y_label, metric=metric, all_class_weights=all_class_weights, vmin=vmin, vmax=vmax, ax=ax)
    #encoded class_weight from 0 to 4 x_ticks need to be different
    # if experiment == "CLASS_WEIGHT":
    #   ax.set_yticklabels(labels=['moderate', 'high', 'balanced', 'extra'], rotation=-5, verticalalignment='baseline',
    #                  horizontalalignment='left')
    return ax, surf

def generate_3d_plots(experiment, model_name, grid_search, metrics, metric_ranges_map, oversample_strategies_map, experiment_ratio_to_name_map, all_class_weights=None):
    fig, axes = plt.subplots(subplot_kw={"projection": "3d"}, ncols=len(metrics), figsize=[12,10])
    for i in range(len(metrics)):
        metric = metrics[i]
        ax, surf = generate_3d_plot(experiment=experiment, model_name=model_name, grid_search=grid_search, metric=metric, metric_ranges_map=metric_ranges_map, oversample_strategies_map=oversample_strategies_map, experiment_ratio_to_name_map=experiment_ratio_to_name_map, ax=axes[i], all_class_weights=all_class_weights)
    # fig.colorbar(surf, shrink=0.2, aspect=2, ax=ax)
        #avoid cutoff
        ax.set_box_aspect(aspect=None, zoom=0.90)
    return fig

import os
#careful this savefig can overwrite existing figures
def savefig_for_overleaf(dataset_name, fig_name, fig):
    dir = os.getcwd()
    fname = dir + "/figures_out/" + dataset_name + "/" + fig_name + ".svg"
    fig.savefig(fname, format='svg', bbox_inches='tight')


