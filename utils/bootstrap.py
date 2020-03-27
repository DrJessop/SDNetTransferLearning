from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt


def bootstrap_auc(y_true, y_pred, ax, nsamples=1000):

    from scipy.interpolate import interp1d

    auc_values = []

    tpr_values = []

    for b in range(nsamples):

        idx = np.random.randint(y_true.shape[0], size=y_true.shape[0])
        y_true_bs = y_true[idx]
        y_pred_bs = y_pred[idx]
        fpr, tpr, _ = roc_curve(y_true_bs, y_pred_bs, drop_intermediate=True)

        if b == 0:
            fpr_interp = fpr

        f = interp1d(fpr, tpr)
        tpr_interp = f(fpr_interp)
        roc_auc = roc_auc_score(y_true_bs, y_pred_bs)
        auc_values.append(roc_auc)
        tpr_values.append(tpr_interp)

    auc_ci = np.percentile(auc_values, (2.5, 97.5))
    auc_mean = np.mean(auc_values)
    tprs_ci = np.percentile(tpr_values, (2.5, 97.5), axis=0)
    tprs_mean = np.mean(tpr_values, axis=0)
    ax.fill_between(fpr_interp, tprs_ci[0], tprs_ci[1], color='k', alpha=0.2, zorder=1, label='95% CI')
    ax.plot(fpr_interp, tprs_mean, color='k', label='AUC: {0:.3f} ({1:.3f}-{2:.3f})'.format(auc_mean, auc_ci[0], auc_ci[1]), linewidth=0.8, zorder=0)
    ax.plot([0, 1], [0, 1], color='crimson', linestyle='--', alpha=1, linewidth=1.5, label='Reference')
    ax.set_xlim([-0.01, 1.00])
    ax.set_ylim([-0.01, 1.01])
    ax.set_ylabel('Sensitivity')
    ax.set_xlabel('1 - Specificity')
    plt.legend(loc="lower right")
    plt.grid(color='k', alpha=0.5)