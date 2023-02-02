import numpy as np
from sklearn.metrics import auc,precision_recall_curve

def get_cindex(dataY, predictData):
    predictData = predictData[:,np.newaxis] - predictData
    predictData = np.float32(predictData==0) * 0.5 + np.float32(predictData>0)

    dataY = dataY[:,np.newaxis] - dataY
    dataY = np.tril(np.float32(dataY>0), 0)

    predictData_sum = np.sum(predictData*dataY)
    targetdata_sum = np.sum(dataY)

    if targetdata_sum==0:
        return 0
    else:
        return predictData_sum/targetdata_sum

def r_squared_error(targetdata_obs,targetdata_pred):
    targetdata_obs = np.array(targetdata_obs)
    targetdata_pred = np.array(targetdata_pred)
    targetdata_obs_mean = [np.mean(targetdata_obs) for y in targetdata_obs]
    targetdata_pred_mean = [np.mean(targetdata_pred) for y in targetdata_pred]

    mult = sum((targetdata_pred - targetdata_pred_mean) * (targetdata_obs - targetdata_obs_mean))
    mult = mult * mult

    targetdata_obs_sq = sum((targetdata_obs - targetdata_obs_mean)*(targetdata_obs - targetdata_obs_mean))
    targetdata_pred_sq = sum((targetdata_pred - targetdata_pred_mean) * (targetdata_pred - targetdata_pred_mean) )

    return mult / float(targetdata_obs_sq * targetdata_pred_sq)

def get_k(targetdata_obs,targetdata_pred):
    targetdata_obs = np.array(targetdata_obs)
    targetdata_pred = np.array(targetdata_pred)

    return sum(targetdata_obs*targetdata_pred) / float(sum(targetdata_pred*targetdata_pred))

def squared_error_zero(targetdata_obs,targetdata_pred):
    k = get_k(targetdata_obs,targetdata_pred)
    targetdata_obs = np.array(targetdata_obs)
    targetdata_pred = np.array(targetdata_pred)
    targetdata_obs_mean = [np.mean(targetdata_obs) for y in targetdata_obs]
    upp = sum((targetdata_obs - (k*targetdata_pred)) * (targetdata_obs - (k* targetdata_pred)))
    down= sum((targetdata_obs - targetdata_obs_mean)*(targetdata_obs - targetdata_obs_mean))

    return 1 - (upp / float(down))

def get_rm2(ys_orig,ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r_randtwo = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r_randtwo*r_randtwo))))

def get_aupr(targetdata_true,targetdata_pred):
    precision, recall, thresholds = precision_recall_curve(targetdata_true,targetdata_pred)
    roc_aupr = auc(recall,precision)
    return roc_aupr
