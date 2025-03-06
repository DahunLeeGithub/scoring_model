import pandas as pd
from app.models.financialData import FinancialData
from sklearn.preprocessing import StandardScaler
import torch

def preprocessing(data: FinancialData) -> torch.Tensor:
    def calculate_altman_z_score(df: pd.DataFrame, period):
        # period should be "T_2_" for two periods ago or "T_1_" for last period
        df["X1"] = (df[f"{period}CurrentAssets"] - df[f"{period}CurrentLiabilities"]) / df[f"{period}TotalAssets"]
        df["X2"] = df[f"{period}UndistributedProfitBS"] / df[f"{period}TotalAssets"]
        df["X3"] = df[f"{period}OperatingIncome"] / df[f"{period}TotalAssets"]
        df["X4"] = df[f"{period}TotalEquity"] / df[f"{period}TotalLiabilities"]
        df["X5"] = df[f"{period}Sales"] / df[f"{period}TotalAssets"]

        # Calculate Altman Z-Score
        df[f"{period}Z-Score"] = 1.2 * df["X1"] + 1.4 * df["X2"] + 3.3 * df["X3"] + 0.6 * df["X4"] + 1.0 * df["X5"]
        df.drop(["X1", "X2", "X3", "X4", "X5"], axis=1, inplace=True)

        return df[f"{period}Z-Score"]
    EPSILON = 1e-8
    origin_data = pd.DataFrame(data)
    data = pd.DataFrame()
    col_text = (
        "IndustryCode,"
        "T_2_CashAndDeposits,T_2_NotesReceivable,T_2_AccountsReceivable,T_2_OtherCurrentAssets,"
        "T_2_CompletedProjectsAccruedIncome,T_2_ConstructionInProgressExpenses,T_2_Inventory,"
        "T_2_OtherCurrentAssets2,T_2_ShortTermLoansReceivable,T_2_CurrentAssets,"
        "T_2_AllowanceForDoubtfulAccounts,T_2_TangibleFixedAssets,T_2_IntangibleFixedAssets,"
        "T_2_InvestmentsAndOtherAssets,T_2_DeferredAssets,T_2_LongTermLoansReceivable,"
        "T_2_FixedAssets,T_2_TotalAssets,T_2_NotesPayable,T_2_AccountsPayable,T_2_ShortTermBorrowings,"
        "T_2_AccruedExpenses,T_2_Provisions,T_2_AdvanceReceivedForConstruction,T_2_AdvanceReceived,"
        "T_2_OtherCurrentLiabilities,T_2_CurrentLiabilities,T_2_LongTermBorrowings,"
        "T_2_OtherNonCurrentLiabilities,T_2_NonCurrentLiabilities,T_2_TotalLiabilities,"
        "T_2_CapitalStock,T_2_Reserves,T_2_UndistributedProfitBS,T_2_TotalEquity,"
        "T_2_TotalLiabilitiesEquity,T_2_AssetLiabilityError,T_2_DiscountedNotes,T_2_EndorsedNotes,"
        "T_2_Sales,T_2_CostOfGoodsSold,T_2_GrossProfit,T_2_SG&AExpenses,T_2_OperatingIncome,"
        "T_2_InterestIncome,T_2_OtherNonOperatingIncome,T_2_TotalNonOperatingIncome,"
        "T_2_InterestExpense,T_2_OtherNonOperatingExpenses,T_2_TotalNonOperatingExpenses,"
        "T_2_OrdinaryIncome,T_2_SpecialGains,T_2_SpecialLosses,T_2_IncomeBeforeTax,T_2_CorporateTax,"
        "T_2_NetIncome,T_2_CarryForwardProfit,T_2_InterimDividends,T_2_UndistributedProfitPL,"
        "T_2_BSPL_Error,T_2_CarryForwardProfitError,T_2_SpecialDepreciationReserve,T_2_Depreciation,"
        "T_1_CashAndDeposits,T_1_NotesReceivable,T_1_AccountsReceivable,T_1_OtherCurrentAssets,"
        "T_1_CompletedProjectsAccruedIncome,T_1_ConstructionInProgressExpenses,T_1_Inventory,"
        "T_1_OtherCurrentAssets2,T_1_ShortTermLoansReceivable,T_1_CurrentAssets,"
        "T_1_AllowanceForDoubtfulAccounts,T_1_TangibleFixedAssets,T_1_IntangibleFixedAssets,"
        "T_1_InvestmentsAndOtherAssets,T_1_DeferredAssets,T_1_LongTermLoansReceivable,"
        "T_1_FixedAssets,T_1_TotalAssets,T_1_NotesPayable,T_1_AccountsPayable,T_1_ShortTermBorrowings,"
        "T_1_AccruedExpenses,T_1_Provisions,T_1_AdvanceReceivedForConstruction,T_1_AdvanceReceived,"
        "T_1_OtherCurrentLiabilities,T_1_CurrentLiabilities,T_1_LongTermBorrowings,"
        "T_1_OtherNonCurrentLiabilities,T_1_NonCurrentLiabilities,T_1_TotalLiabilities,"
        "T_1_CapitalStock,T_1_Reserves,T_1_UndistributedProfitBS,T_1_TotalEquity,"
        "T_1_TotalLiabilitiesEquity,T_1_AssetLiabilityError,T_1_DiscountedNotes,T_1_EndorsedNotes,"
        "T_1_Sales,T_1_CostOfGoodsSold,T_1_GrossProfit,T_1_SG&AExpenses,T_1_OperatingIncome,"
        "T_1_InterestIncome,T_1_OtherNonOperatingIncome,T_1_TotalNonOperatingIncome,"
        "T_1_InterestExpense,T_1_OtherNonOperatingExpenses,T_1_TotalNonOperatingExpenses,"
        "T_1_OrdinaryIncome,T_1_SpecialGains,T_1_SpecialLosses,T_1_IncomeBeforeTax,T_1_CorporateTax,"
        "T_1_NetIncome,T_1_CarryForwardProfit,T_1_InterimDividends,T_1_UndistributedProfitPL,"
        "T_1_BSPL_Error,T_1_CarryForwardProfitError,T_1_SpecialDepreciationReserve,T_1_Depreciation,"
        "T_2_FiscalPeriod,T_1_FiscalPeriod"
    )
    cols = col_text.split(',')
    # Remove columns that are not needed
    cols_remove = ['FiscalPeriod']
    cols_remove = [f'T_2_{col}' for col in cols_remove] + [f'T_1_{col}' for col in cols_remove]
    cols = [col for col in cols if col not in cols_remove]

    data = origin_data[cols].copy()
    data = data.replace(',', '', regex=True)
    data = data.apply(pd.to_numeric, errors='coerce')

    # One-hot encode the industry code
    dummies = pd.get_dummies(data['IndustryCode'], prefix='IndustryCode', drop_first=True)
    data = pd.concat([data.drop('IndustryCode', axis=1), dummies], axis=1)

    # Calculate various financial ratios
    data['T_2_CurrentRatio'] = data['T_2_CurrentAssets'] / data['T_2_CurrentLiabilities'].replace(0, EPSILON)
    data['T_1_CurrentRatio'] = data['T_1_CurrentAssets'] / data['T_1_CurrentLiabilities'].replace(0, EPSILON)

    data['T_2_QuickRatio'] = (data['T_2_CurrentAssets'] - data['T_2_Inventory']) / data['T_2_CurrentLiabilities'].replace(0, EPSILON)
    data['T_1_QuickRatio'] = (data['T_1_CurrentAssets'] - data['T_1_Inventory']) / data['T_1_CurrentLiabilities'].replace(0, EPSILON)

    data['T_2_CashRatio'] = data['T_2_CashAndDeposits'] / data['T_2_CurrentAssets'].replace(0, EPSILON)
    data['T_1_CashRatio'] = data['T_1_CashAndDeposits'] / data['T_1_CurrentAssets'].replace(0, EPSILON)

    data['T_2_DebtRatio'] = data['T_2_TotalLiabilities'] / data['T_2_TotalAssets'].replace(0, EPSILON)
    data['T_1_DebtRatio'] = data['T_1_TotalLiabilities'] / data['T_1_TotalAssets'].replace(0, EPSILON)

    data['T_2_InterestCoverageRatio'] = data['T_2_OperatingIncome'] / data['T_2_InterestExpense'].replace(0, EPSILON)
    data['T_1_InterestCoverageRatio'] = data['T_1_OperatingIncome'] / data['T_1_InterestExpense'].replace(0, EPSILON)

    data['T_2_EquityRatio'] = data['T_2_TotalLiabilities'] / data['T_2_TotalEquity'].replace(0, EPSILON)
    data['T_1_EquityRatio'] = data['T_1_TotalLiabilities'] / data['T_1_TotalEquity'].replace(0, EPSILON)

    data['T_2_GrossMargin'] = data['T_2_GrossProfit'] / data['T_2_Sales'].replace(0, EPSILON)
    data['T_1_GrossMargin'] = data['T_1_GrossProfit'] / data['T_1_Sales'].replace(0, EPSILON)

    data['T_2_OperatingMargin'] = data['T_2_OperatingIncome'] / data['T_2_Sales'].replace(0, EPSILON)
    data['T_1_OperatingMargin'] = data['T_1_OperatingIncome'] / data['T_1_Sales'].replace(0, EPSILON)

    data['T_2_NetMargin'] = data['T_2_NetIncome'] / data['T_2_Sales'].replace(0, EPSILON)
    data['T_1_NetMargin'] = data['T_1_NetIncome'] / data['T_1_Sales'].replace(0, EPSILON)

    data['T_2_TotalAssetTurnover'] = data['T_2_Sales'] / data['T_2_TotalAssets'].replace(0, EPSILON)
    data['T_1_TotalAssetTurnover'] = data['T_1_Sales'] / data['T_1_TotalAssets'].replace(0, EPSILON)

    data['T_2_ReceivablesTurnover'] = data['T_2_Sales'] / data['T_2_AccountsReceivable'].replace(0, EPSILON)
    data['T_1_ReceivablesTurnover'] = data['T_1_Sales'] / data['T_1_AccountsReceivable'].replace(0, EPSILON)

    data['T_2_InventoryTurnover'] = data['T_2_Sales'] / data['T_2_Inventory'].replace(0, EPSILON)
    data['T_1_InventoryTurnover'] = data['T_1_Sales'] / data['T_1_Inventory'].replace(0, EPSILON)

    data['ProfitVolatility'] = abs(data['T_1_OperatingIncome'] - data['T_2_OperatingIncome']) / data['T_2_OperatingIncome'].replace(0, EPSILON)

    data['T_2_NegativeCashFlow'] = (data['T_2_OperatingIncome'] < 0) | (data['T_2_CashAndDeposits'] < data['T_2_ShortTermBorrowings'])
    data['T_1_NegativeCashFlow'] = (data['T_1_OperatingIncome'] < 0) | (data['T_1_CashAndDeposits'] < data['T_1_ShortTermBorrowings'])

    data['T_1_CumulativeLoss'] = data['T_1_UndistributedProfitBS'] < 0

    data['SalesGrowthRate'] = (data['T_1_Sales'] - data['T_2_Sales']) / data['T_2_Sales'].replace(0, EPSILON)
    data['COGSGrowthRate'] = (data['T_1_CostOfGoodsSold'] - data['T_2_CostOfGoodsSold']) / data['T_2_CostOfGoodsSold'].replace(0, EPSILON)
    data['OperatingIncomeGrowthRate'] = (data['T_1_OperatingIncome'] - data['T_2_OperatingIncome']) / data['T_2_OperatingIncome'].replace(0, EPSILON)
    data['DebtGrowthRate'] = (data['T_1_TotalLiabilities'] - data['T_2_TotalLiabilities']) / data['T_2_TotalLiabilities'].replace(0, EPSILON)

    data['T_2_Z-Score'] = calculate_altman_z_score(data, 'T_2_')
    data['T_1_Z-Score'] = calculate_altman_z_score(data, 'T_1_')

    # Specify columns for which change rates are needed (excluding some columns)
    before_cols = [col for col in cols if col.startswith('T_1_')]
    no_change_cols = ['AssetGrowthRate', 'FSRI', 'T_1_CumulativeLoss', 'SalesGrowthRate', 'COGSGrowthRate', 'OperatingIncomeGrowthRate', 'DebtGrowthRate']
    for col in before_cols:
        if col in no_change_cols:
            continue
        data[col[4:] + '_ChangeRate'] = (data[col] - data[col.replace('T_1_', 'T_2_', 1)]) / data[col.replace('T_1_', 'T_2_', 1)]

    # Standardize the features
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    tensor = torch.tensor(data, dtype=torch.float32)

    lower_bound = torch.quantile(tensor, 0.01)
    upper_bound = torch.quantile(tensor, 0.99)

    tensor = torch.clamp(tensor, lower_bound, upper_bound)

    return tensor
