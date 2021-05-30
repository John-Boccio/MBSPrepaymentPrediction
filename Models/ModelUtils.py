import pandas as pd
import matplotlib.pyplot as plt
import datetime


def plot_cpr(testDf, pred, path):
    df = testDf[['report_month', 'zero_balance_removal_UPB', 'prepay', 'current_UPB']]
    plotDf = pd.concat([testDf[['report_month', 'zero_balance_removal_UPB', 'prepay', 'current_UPB']], pd.DataFrame(pred)], axis=1)
    plotDf.columns = ['report_month', 'zero_balance_removal_UPB', 'prepay', 'current_UPB', 'pred_prepay']
    plotDf['balance'] = plotDf.apply(
        lambda row: row['current_UPB'] if row['prepay'] == 0 else row['zero_balance_removal_UPB'], axis=1)

    plotDf['actual'] = plotDf.balance * plotDf.prepay
    plotDf['projected'] = plotDf.balance * plotDf.pred_prepay
    plotDf = plotDf.groupby('report_month').agg({'actual': 'sum', 'projected': 'sum', 'balance': 'sum'}).reset_index()

    plotDf['actual_smm'] = plotDf.actual / plotDf.balance
    plotDf['projected_smm'] = plotDf.projected / plotDf.balance
    plotDf['actual_cpr'] = 1 - (1 - plotDf.actual_smm) ** 12
    plotDf['projected_cpr'] = 1 - (1 - plotDf.projected_smm) ** 12
    plotDf['report_month'] = plotDf['report_month'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y%m'))

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('report month')
    ax1.set_ylabel('cpr')
    ax1.plot(plotDf.report_month, plotDf.actual_cpr, color='red', label='actual CPR')
    ax1.plot(plotDf.report_month, plotDf.projected_cpr, color='blue', label='projected CPR')
    # ax1.set_ylim(0, 0.8)
    ax1.legend()
    ax2 = ax1.twinx()
    ax2.set_ylabel('balance')
    ax2.plot(plotDf.report_month, plotDf.balance, color='black', label='balance')
    # ax2.set_ylim(0, 200000000)
    ax2.legend()
    plt.show()
    fig.savefig(path)