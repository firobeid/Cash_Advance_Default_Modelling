import psutil
def memory()->str:
    '''
    Track data allocation memory
    '''
    return print('used: {}% free: {:.2f}GB'.format(psutil.virtual_memory().percent, float(psutil.virtual_memory().free)/1024**3))#@ 

def pd_events(group):
    try:
        # group = group.dropna()
        group[score] = group[score].round(0)
        length = len(group.loc[group[target_name].notna()])
        events = group.loc[~(group[target_name].isna()), target_name].sum()
        non_events = (1 - group.loc[~(group[target_name].isna()),target_name]).sum()
        cols = ['N','NUM_EVENTS', 'NUM_NON_EVENTS']
        return pd.Series([length,events,non_events], index=cols)
    except:
        return np.nan

def ks_scoring(group):
    '''
    # ks_scoring(df.loc[df.Partition_Column == 'V'].groupby([score]).apply(pd_events))
    '''
    try:
        def segment(group):
            return max(abs((group["NUM_EVENTS"].cumsum()/group["NUM_EVENTS"].sum()) - (group["NUM_NON_EVENTS"].cumsum()/group["NUM_NON_EVENTS"].sum())))
        ks = segment(group)
        # N = sum(group["N"])
        # cols = ["KS","COUNT"]
        return ks
    except:
        return np.nan
        
def pd_good_bad(group):
    try:
        # group = group.dropna()
        group[score] = group[score].round(0)
        length = len(group.loc[group[target_name].notna()])
        pd70_bad = group.loc[~(group[target_name].isna()) & (group[target_name] == 1.0), target_name].count()
        if pd70_bad == 0: pd70_bad = 0
        pd70_good = group.loc[~(group[target_name].isna()) & (group[target_name] == 0.0), target_name].count()
        if pd70_good == 0: pd70_good = 0
        cols = ['N','PD70_BAD', 'PD70_GOOD']
        return pd.Series([length,pd70_bad,pd70_good], index=cols)
    except Exception as e:
        print(e)
        return np.nan

def AUC_T(group):
    def trapezoidal_rule(x,y):
        slice1 = [slice(None)]
        slice2 = [slice(None)]
        slice1[-1] = slice(1,None)
        slice2[-1] = slice(None, -1)
        return (np.diff(x, axis = -1) * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0).sum(axis = -1)
    try:
        TPR = group["PD70_BAD"].cumsum() / group["PD70_BAD"].sum()
        FPR = group["PD70_GOOD"].cumsum() / group["PD70_GOOD"].sum()
        auc = trapezoidal_rule(FPR.to_numpy(),TPR.to_numpy())
        # N = sum(group["N"])
        # cols = ["AUC","Count"]
        # return trapezoidal_rule(FPR.to_numpy(),TPR.to_numpy())
        return auc
    except:
        return np.nan

def calculate_external_scores(score_df : pd.DataFrame, 
                              score : str, 
                              target_name_ : str):
    global target_name
    target_name = target_name_
    holdout_set = score_df.copy()#.rename(columns = {target_name :"TARGET"}) #target_name is defined above in set target section
    if holdout_set[score].max() < 1 :
        temp = holdout_set[holdout_set[score].notna()].copy()
        auc_score = auc_point_estimate(temp[target_name], temp[score])
        ks_score = ks_point_estimate(temp[target_name], temp[score])

    else:
        holdout_set[score] = holdout_set[score].round(0)
        auc_score = AUC_T(holdout_set.groupby([score]).apply(pd_good_bad).reset_index())
        ks_score = ks_scoring(holdout_set.groupby([score]).apply(pd_events).reset_index())
    return ks_score, auc_score

###############
#Single Scores#
###############
def metrics_slices_ks(x, score):
    # x = x.dropna()
    # x[score] = x[score].fillna(-1)
    if x[score].max() < 1 :
        results =  pd.Series({'Observations':int(len(x)), 
                          "Inter_GRP_BadRate":round(x[target_name].mean(), 4),
                          "Sample_Pct_Size":round(len(x[target_name]) / len(temp), 4),
                          "%s KS"%score: round(ks_point_estimate(x[target_name], x[score]), 4)})
    else: 
        results =  pd.Series({'Observations':len(x), 
                          "Inter_Group_BadRate":round(x[target_name].mean(), 4),                       
                      "%s KS"%score: round(ks_scoring(x.groupby([score]).apply(pd_events)), 4)})
    return results

def metrics_slices_auc(x, score):

    if x[score].max() < 1 :
        results =  pd.Series({'Observations':int(len(x)), 
                          "Inter_GRP_BadRate":round(x[target_name].mean(), 4),
                          "Sample_Pct_Size":round(len(x[target_name]) / len(temp), 4),
                          "%s AUC"%score:round(auc_point_estimate(x[target_name], x[score]), 4)})
    else:
        results = pd.Series({'Observations':len(x), 
                          "Inter_Group_BadRate":round(x[target_name].mean(), 4),
                      "%s AUC"%score: round(AUC_T(x.groupby([score]).apply(pd_good_bad)), 4)})
    return results


###############
#Multi- Scores#
###############
def metrics_slices_auc_(x, scores):
    global score, target_name
    # x = x.loc[~x[score].isna()]
    results = {'Observations':len(x), 
               "Inter_GRP_BadRate":round(x[target_name].mean(), 4),
               "Sample_Pct_Size":round(len(x[target_name]) / len(temp), 4)}
    for score in scores:
        x[score] = x[score].fillna(-1)
        if x[score].max() < 1 :
            results.update({
                "%s AUC"%score:round(auc_point_estimate(x[target_name], x[score]), 4)
            })
            # return pd.Series(results)
        else:
            results.update({
                    "%s AUC"%score: round(AUC_T(x.groupby([score]).apply(pd_good_bad)), 4)
                })
    return pd.Series(results)
  

def metrics_slices_ks_(x, scores):
    global score, target_name
    results = {'Observations': int(len(x)), 
               "Inter_GRP_BadRate": round(x[target_name].mean(), 4),
               "Sample_Pct_Size": round(len(x[target_name]) / len(temp), 4)}
    for score in scores:
        # x[score] = x[score].dropna()
        if x[score].max() < 1 :
                results.update({
                    "%s KS" % score: round(ks_point_estimate(x[target_name], x[score]), 4)
                })
                # return pd.Series(results)
        else: 
            # results.update({
            #     "%s KS" % score: round(ks_scoring(x.groupby([score]).apply(pd_events)), 4)
            #     })
            results.update({
                "%s KS" % score: round(calculate_ks(x[target_name], x[score]), 4)
                })
    return pd.Series(results)

def calculate_ks(y,x):
    """
    Calculate the Kolmogorov-Smirnov (KS) statistic. Works best for 3 digit scores.

    Args:
        x: Predicted scores.
        y: Actual labels.

    Returns:
        ks_statistic: The KS statistic.
    """
    
    # Sort the predicted scores and actual labels in ascending order of scores
    sorted_indices = np.argsort(np.array(x))
    sorted_x = np.array(x)[sorted_indices]
    sorted_y = np.array(y)[sorted_indices]

    # Calculate the cumulative distribution function (CDF) of the positive class (label=1)
    cdf_positive = np.cumsum(sorted_y) / np.sum(sorted_y)

    # Calculate the cumulative distribution function (CDF) of the negative class (label=0)
    cdf_negative = np.cumsum(1 - sorted_y) / np.sum(1 - sorted_y)

    # Calculate the KS statistic as the maximum absolute difference between the two CDFs
    ks_statistic = np.max(np.abs(cdf_positive - cdf_negative))

    return ks_statistic

def auc_point_estimate(y_real, y_proba):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_real, y_proba)



def gainsTable_non_proba(x, y, breaks=10, cumlDown=True, total=True):

    # Create DataFrame
    df = pd.DataFrame({'score': x, 'response': y})
    df = df[(df['response'] >= 0) & (df['response'] <= 1)]

    # Check if breaks is an integer
    if isinstance(breaks, int):
        breaks = breaks

    # Calculate quantiles based on equal frequency
    quantiles = np.linspace(0, 100, breaks + 1)
    thresholds = np.percentile(df['score'], quantiles)
    thresholds[0] = -np.inf  # Set the first threshold to -inf
    thresholds[-1] = np.inf  # Set the last threshold to inf

    # Bin X based on thresholds
    df['Score_Range'] = pd.cut(df['score'], bins=thresholds, include_lowest=True)

    # Group by Score_Range
    gt = df.groupby('Score_Range').agg(
        Count=('score', 'size'),
        Events=('response', 'sum'),
        Non_Events=('response', lambda x: (1 - x).sum()),
        Mid_Point=('score', 'median')
    ).reset_index()

    # Sort by Mid_Point
    gt = gt.sort_values('Mid_Point', ascending=cumlDown)

    # Calculate metrics
    gt['% Total'] = gt['Count'] / gt['Count'].sum()
    gt['Cuml. % Total'] = gt['% Total'].cumsum()
    gt['Interval_Rate'] = gt['Events'] / gt['Count']
    gt['% Events'] = gt['Events'] / gt['Events'].sum()
    gt['Cuml. % Events'] = gt['% Events'].cumsum()
    gt['% Non_Events'] = gt['Non_Events'] / gt['Non_Events'].sum()
    gt['Cuml. % Non_Events'] = gt['% Non_Events'].cumsum()
    gt['KS'] = np.abs(gt['Cuml. % Events'] - gt['Cuml. % Non_Events'])

    # Select columns
    gt = gt[['Score_Range', 'Count', '% Total', 'Cuml. % Total', 'Interval_Rate',
             'Events', '% Events', 'Cuml. % Events',
             'Non_Events', '% Non_Events', 'Cuml. % Non_Events',
             'Mid_Point', 'KS']]

    if total:
        total_row = pd.DataFrame({
            'Score_Range': 'Total',
            'Count': len(df),
            '% Total': 1,
            'Cuml. % Total': 1,
            'Interval_Rate': df['response'].sum() / len(df),
            'Events': df['response'].sum(),
            '% Events': 1,
            'Cuml. % Events': 1,
            'Non_Events': (1 - df['response']).sum(),
            '% Non_Events': 1,
            'Cuml. % Non_Events': 1,
            'Mid_Point': df['score'].median(),
            'KS': gt['KS'].max()
        }, index=[0])
        gt = pd.concat([gt, total_row], ignore_index=True)

    return gt