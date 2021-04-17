from recommenders.reco_utils.evaluation.python_evaluation import \
    (rmse, mae, rsquared, exp_var, map_at_k, ndcg_at_k, precision_at_k, recall_at_k)


def eval_pointwise(true_df, pred_scores):
    eval_rmse = rmse(true_df.data, pred_scores, col_user=true_df.user_col, col_item=true_df.item_col,
                col_rating=true_df.score_col)
    eval_mae = mae(true_df.data, pred_scores, col_user=true_df.user_col, col_item=true_df.item_col,
                col_rating=true_df.score_col)
    eval_rsquared = rsquared(true_df.data, pred_scores, col_user=true_df.user_col, col_item=true_df.item_col,
                col_rating=true_df.score_col)
    eval_exp_var = exp_var(true_df.data, pred_scores, col_user=true_df.user_col, col_item=true_df.item_col,
                col_rating=true_df.score_col)
    return {
        'rmse': eval_rmse,
        'mae': eval_mae,
        'r2': eval_rsquared,
        'exp_var': eval_exp_var,
    }


def eval_top(true_df, pred_top, k):
    eval_map = map_at_k(true_df.data, pred_top, col_prediction='prediction', k=k, col_user=true_df.user_col,
                col_item=true_df.item_col, col_rating=true_df.score_col)
    eval_ndcg = ndcg_at_k(true_df.data, pred_top, col_prediction='prediction', k=k,
                col_user=true_df.user_col, col_item=true_df.item_col, col_rating=true_df.score_col)
    eval_precision = precision_at_k(true_df.data, pred_top, col_prediction='prediction', k=k,
                col_user=true_df.user_col, col_item=true_df.item_col, col_rating=true_df.score_col)
    eval_recall = recall_at_k(true_df.data, pred_top, col_prediction='prediction', k=k,
                col_user=true_df.user_col, col_item=true_df.item_col, col_rating=true_df.score_col)
    return {
        'map_' + str(k): eval_map,
        'ndcg_' + str(k): eval_ndcg,
        'precision_' + str(k): eval_precision,
        'recall_' + str(k): eval_recall,
    }
