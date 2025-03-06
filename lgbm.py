import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from lightgbm.basic import Booster

def calculate_altman_z_score(df: pd.DataFrame, period):
    # period should be 'T-2_' or 'T-1_'
    df["X1"] = (df[f"{period}CurrentAssets"] - df[f"{period}CurrentLiabilities"]) / df[f"{period}TotalAssets"]
    df["X2"] = df[f"{period}UndistributedProfitBS"] / df[f"{period}TotalAssets"]
    df["X3"] = df[f"{period}OperatingIncome"] / df[f"{period}TotalAssets"]
    df["X4"] = df[f"{period}TotalEquity"] / df[f"{period}TotalLiabilities"]
    df["X5"] = df[f"{period}Sales"] / df[f"{period}TotalAssets"]

    # Calculate Altman Z-Score
    df[f"{period}Z-Score"] = 1.2 * df["X1"] + 1.4 * df["X2"] + 3.3 * df["X3"] + 0.6 * df["X4"] + 1.0 * df["X5"]
    df.drop(["X1", "X2", "X3", "X4", "X5"], axis=1, inplace=True)

    return df[f"{period}Z-Score"]

def calculate_fsri(df: pd.DataFrame):
    w1 = 0.3
    w2 = 0.2
    w3 = 0.25
    w4 = 0.15
    w5 = 0.1
    df['FSRI'] = (df["T-2_OperatingIncome"] / df["T-2_TotalAssets"]) * w1 \
               + (df["T-2_GrossProfit"] / df["T-2_Sales"]) * w2 \
               + (df["T-2_NetIncome"] / df["T-2_TotalEquity"]) * w3 \
               + (df["T-2_CurrentAssets"] / df["T-2_CurrentLiabilities"]) * w4 \
               + (df["T-2_Sales"] - df["T-1_Sales"]) / df["T-1_Sales"] * w5
    return df

def load_data(path: str) -> pd.DataFrame:
    origin_data = pd.read_csv(path)
    origin_data = origin_data.replace(',', '', regex=True)

    # List of columns in English. Note: Adjust these names as needed.
    col_text = (
        "IndustryCode,"
        "T-2_CashAndDeposits,"
        "T-2_NotesReceivable,"
        "T-2_AccountsReceivable,"
        "T-2_OtherShortTermAssets,"
        "T-2_AccruedIncomeFromCompletedProjects,"
        "T-2_ConstructionInProgressExpenses,"
        "T-2_Inventory,"
        "T-2_OtherCurrentAssets,"
        "T-2_ShortTermLoansReceivable,"
        "T-2_CurrentAssets,"
        "T-2_AllowanceForDoubtfulAccounts,"
        "T-2_TangibleFixedAssets,"
        "T-2_IntangibleFixedAssets,"
        "T-2_InvestmentsAndOtherAssets,"
        "T-2_DeferredAssets,"
        "T-2_LongTermLoansReceivable,"
        "T-2_FixedAssets,"
        "T-2_TotalAssets,"
        "T-2_NotesPayable,"
        "T-2_AccountsPayable,"
        "T-2_ShortTermBorrowings,"
        "T-2_AccruedExpenses,"
        "T-2_Provisions,"
        "T-2_AdvanceReceivedForConstruction,"
        "T-2_AdvanceReceived,"
        "T-2_OtherCurrentLiabilities,"
        "T-2_CurrentLiabilities,"
        "T-2_LongTermBorrowings,"
        "T-2_OtherNonCurrentLiabilities,"
        "T-2_NonCurrentLiabilities,"
        "T-2_TotalLiabilities,"
        "T-2_CapitalStock,"
        "T-2_Reserves,"
        "T-2_UndistributedProfitBS,"
        "T-2_TotalEquity,"
        "T-2_TotalLiabilitiesEquity,"
        "T-2_AssetLiabilityError,"
        "T-2_DiscountedNotes,"
        "T-2_EndorsedNotes,"
        "T-2_Sales,"
        "T-2_CostOfGoodsSold,"
        "T-2_GrossProfit,"
        "T-2_SG&AExpenses,"
        "T-2_OperatingIncome,"
        "T-2_InterestIncome,"
        "T-2_OtherNonOperatingIncome,"
        "T-2_TotalNonOperatingIncome,"
        "T-2_InterestExpense,"
        "T-2_OtherNonOperatingExpenses,"
        "T-2_TotalNonOperatingExpenses,"
        "T-2_OrdinaryIncome,"
        "T-2_SpecialGains,"
        "T-2_SpecialLosses,"
        "T-2_IncomeBeforeTax,"
        "T-2_CorporateTax,"
        "T-2_NetIncome,"
        "T-2_CarryForwardProfit,"
        "T-2_InterimDividendsEtc,"
        "T-2_UndistributedProfitPL,"
        "T-2_BSPL_Error,"
        "T-2_CarryForwardProfitError,"
        "T-2_SpecialDepreciationReserve,"
        "T-2_Depreciation,"
        "T-1_CashAndDeposits,"
        "T-1_NotesReceivable,"
        "T-1_AccountsReceivable,"
        "T-1_OtherShortTermAssets,"
        "T-1_AccruedIncomeFromCompletedProjects,"
        "T-1_ConstructionInProgressExpenses,"
        "T-1_Inventory,"
        "T-1_OtherCurrentAssets,"
        "T-1_ShortTermLoansReceivable,"
        "T-1_CurrentAssets,"
        "T-1_AllowanceForDoubtfulAccounts,"
        "T-1_TangibleFixedAssets,"
        "T-1_IntangibleFixedAssets,"
        "T-1_InvestmentsAndOtherAssets,"
        "T-1_DeferredAssets,"
        "T-1_LongTermLoansReceivable,"
        "T-1_FixedAssets,"
        "T-1_TotalAssets,"
        "T-1_NotesPayable,"
        "T-1_AccountsPayable,"
        "T-1_ShortTermBorrowings,"
        "T-1_AccruedExpenses,"
        "T-1_Provisions,"
        "T-1_AdvanceReceivedForConstruction,"
        "T-1_AdvanceReceived,"
        "T-1_OtherCurrentLiabilities,"
        "T-1_CurrentLiabilities,"
        "T-1_LongTermBorrowings,"
        "T-1_OtherNonCurrentLiabilities,"
        "T-1_NonCurrentLiabilities,"
        "T-1_TotalLiabilities,"
        "T-1_CapitalStock,"
        "T-1_Reserves,"
        "T-1_UndistributedProfitBS,"
        "T-1_TotalEquity,"
        "T-1_TotalLiabilitiesEquity,"
        "T-1_AssetLiabilityError,"
        "T-1_DiscountedNotes,"
        "T-1_EndorsedNotes,"
        "T-1_Sales,"
        "T-1_CostOfGoodsSold,"
        "T-1_GrossProfit,"
        "T-1_SG&AExpenses,"
        "T-1_OperatingIncome,"
        "T-1_InterestIncome,"
        "T-1_OtherNonOperatingIncome,"
        "T-1_TotalNonOperatingIncome,"
        "T-1_InterestExpense,"
        "T-1_OtherNonOperatingExpenses,"
        "T-1_TotalNonOperatingExpenses,"
        "T-1_OrdinaryIncome,"
        "T-1_SpecialGains,"
        "T-1_SpecialLosses,"
        "T-1_IncomeBeforeTax,"
        "T-1_CorporateTax,"
        "T-1_NetIncome,"
        "T-1_CarryForwardProfit,"
        "T-1_InterimDividendsEtc,"
        "T-1_UndistributedProfitPL,"
        "T-1_BSPL_Error,"
        "T-1_CarryForwardProfitError,"
        "T-1_SpecialDepreciationReserve,"
        "T-1_Depreciation,"
        "T-2_FiscalPeriod,"
        "T-1_FiscalPeriod"
    )
    cols = col_text.split(',')

    # Exclude columns not needed
    cols_remove = ['FiscalPeriod']  # if any column is not used
    cols = [col for col in cols if col not in cols_remove]
    data = origin_data[cols].copy()
    data = data.apply(pd.to_numeric, errors='coerce')

    for col in cols:
        # Missing value processing: fill from the corresponding period if one is missing.
        if col.startswith('T-1_'):
            ref_col = col.replace('T-1_', 'T-2_', 1)
        elif col.startswith('T-2_'):
            ref_col = col.replace('T-2_', 'T-1_', 1)
        else:
            continue

        # If both corresponding columns are missing, fill with the mean.
        if data[col] is None and data[ref_col] is None:
            data[col] = data[col].fillna(data[col].mean())
        # Otherwise, fill missing values with the value from the corresponding column.
        data[col] = data[col].fillna(data[ref_col])

    data['IndustryCode'] = data['IndustryCode'].astype('category')

    # Add Altman Z-Score for both periods
    data = calculate_altman_z_score(data, 'T-2_')
    data = calculate_altman_z_score(data, 'T-1_')

    # Add Asset Growth Rate
    data['AssetGrowthRate'] = (data['T-1_TotalAssets'] - data['T-2_TotalAssets']) / data['T-2_TotalAssets']

    # Add current ratio
    data['T-1_CurrentRatio'] = data['T-1_CurrentAssets'] / data['T-1_CurrentLiabilities']
    data['T-2_CurrentRatio'] = data['T-2_CurrentAssets'] / data['T-2_CurrentLiabilities']

    # Add DSO (Days Sales Outstanding)
    data['T-1_DSO'] = data['T-1_AccountsReceivable'] / data['T-1_Sales'] * 365
    data['T-2_DSO'] = data['T-2_AccountsReceivable'] / data['T-2_Sales'] * 365

    # Add cashflow to debt ratio
    data['T-1_CashFlowToDebt'] = data['T-1_CashAndDeposits'] / data['T-1_TotalLiabilities']
    data['T-2_CashFlowToDebt'] = data['T-2_CashAndDeposits'] / data['T-2_TotalLiabilities']

    # Add debt to asset ratio
    data['T-1_DebtToAsset'] = data['T-1_TotalLiabilities'] / data['T-1_TotalAssets']
    data['T-2_DebtToAsset'] = data['T-2_TotalLiabilities'] / data['T-2_TotalAssets']

    # Add debt to equity ratio (here using TotalLiabilities/TotalEquity)
    data['T-1_DebtToEquity'] = data['T-1_TotalLiabilities'] / data['T-1_TotalEquity']
    data['T-2_DebtToEquity'] = data['T-2_TotalLiabilities'] / data['T-2_TotalEquity']

    # Add Financial Distress
    data['T-1_FinancialDistress'] = (data['T-1_TotalLiabilities'] - data['T-1_TotalAssets']) / data['T-1_TotalAssets']
    data['T-2_FinancialDistress'] = (data['T-2_TotalLiabilities'] - data['T-2_TotalAssets']) / data['T-2_TotalAssets']

    # Add Growth Profit Margin
    data['T-1_GrossProfitMargin'] = data['T-1_GrossProfit'] / data['T-1_Sales']
    data['T-2_GrossProfitMargin'] = data['T-2_GrossProfit'] / data['T-2_Sales']

    # Add Insolvency Probability Index (using InterestExpense / TotalLiabilities)
    data['T-1_InterestExpenseRate'] = data['T-1_InterestExpense'] / data['T-1_TotalLiabilities']
    data['T-2_InterestExpenseRate'] = data['T-2_InterestExpense'] / data['T-2_TotalLiabilities']

    # Add Interest Burden Ratio
    data['T-1_InterestBurden'] = data['T-1_InterestExpense'] / data['T-1_OperatingIncome']
    data['T-2_InterestBurden'] = data['T-2_InterestExpense'] / data['T-2_OperatingIncome']

    # Add Interest Coverage Ratio
    data['T-1_InterestCoverage'] = (data['T-1_OperatingIncome'] + data['T-1_TotalNonOperatingIncome']) / data['T-1_InterestExpense']
    data['T-2_InterestCoverage'] = (data['T-2_OperatingIncome'] + data['T-2_TotalNonOperatingIncome']) / data['T-2_InterestExpense']

    # Add FSRI
    data = calculate_fsri(data)

    # Add Change Rates for T-1 columns (excluding AssetGrowthRate and FSRI)
    before_cols = [col for col in cols if col.startswith('T-1_')]
    no_change_cols = ['AssetGrowthRate', 'FSRI']
    for col in before_cols:
        if col in no_change_cols:
            continue
        # Create a new column name by removing the "T-1_" prefix and appending "_ChangeRate"
        new_col = col[4:] + '_ChangeRate'
        data[new_col] = (data[col] - data[col.replace('T-1_', 'T-2_', 1)]) / data[col.replace('T-1_', 'T-2_', 1)]

    # Drop the original detailed numerical columns except for IndustryCode
    data = data.drop(list(set(cols) - {'IndustryCode'}), axis=1)
    return data

def load_train_data(path: str) -> pd.DataFrame:
    origin_data = pd.read_csv(path)
    data = load_data('test_qualifiers.csv')
    data['AccidentFlag'] = origin_data['AccidentFlag']  # Ensure your CSV uses 'AccidentFlag'
    data['AccidentFlag'] = data['AccidentFlag'].astype('bool')
    data['AccidentFlag'] = data['AccidentFlag'].fillna(0)
    return data

train = load_train_data('train_qualifiers.csv')

X_train, X_test, y_train, y_test = train_test_split(
    train.drop('AccidentFlag', axis=1),
    train['AccidentFlag'],
    test_size=0.2,
    random_state=0
)

# Specify categorical features among the explanatory variables
categorical_features = ['IndustryCode']

# Create the LightGBM datasets
lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, categorical_feature=categorical_features)

params = {
    # Binary classification problem
    'objective': 'binary',
    # Aim to maximize AUC
    'metric': 'auc',
    'early_stopping_round': 50
}

model: Booster = lgb.train(
    params,
    lgb_train,
    valid_sets=lgb_eval,
    num_boost_round=1000  # Maximum iterations
)

model.save_model('model.txt')

y_pred = model.predict(X_test, num_iteration=model.best_iteration)

auc = roc_auc_score(y_test, y_pred)
print(train.corrwith(train['AccidentFlag']).abs().sort_values(ascending=False))
print(auc)

test = load_data('test_qualifiers.csv')
submit_df = pd.read_csv('sample_submission.csv')
submit_df['AccidentFlag'] = model.predict(test, num_iteration=model.best_iteration)
submit_df.to_csv('submit.csv', index=False)
