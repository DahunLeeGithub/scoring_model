import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

pd.set_option('future.no_silent_downcasting', True)

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

def preprocessing(data: pd.DataFrame, length: int) -> torch.Tensor:
    EPSILON = 1e-8
    origin_data = data.copy()
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

    # Remove columns with nearly zero standard deviation
    data = data.loc[:, data.std() > EPSILON]

    # Split data into training and test sets based on provided length
    train = data.iloc[:length]
    test = data.iloc[length:]
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")  # Debug output

    # Standardize the features
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    train_tensor = torch.tensor(train, dtype=torch.float32)
    test_tensor = torch.tensor(test, dtype=torch.float32)

    train_lower_bound = torch.quantile(train_tensor, 0.01)
    train_upper_bound = torch.quantile(train_tensor, 0.99)

    test_lower_bound = torch.quantile(test_tensor, 0.01)
    test_upper_bound = torch.quantile(test_tensor, 0.99)

    train_tensor = torch.clamp(train_tensor, train_lower_bound, train_upper_bound)
    test_tensor = torch.clamp(test_tensor, test_lower_bound, test_upper_bound)

    print(train_tensor.shape)
    print(test_tensor.shape)
    return (train_tensor, test_tensor)

# Read and prepare the data
data = pd.read_csv('train_qualifiers.csv', low_memory=False)
data = data.dropna()
test = pd.read_csv('test_qualifiers.csv', low_memory=False)
test = test.fillna(0)

combined_data = pd.concat([data, test], axis=0)
train_features, test_features = preprocessing(combined_data, len(data))

# Label processing
labels = torch.tensor(data['AccidentFlag'].values, dtype=torch.float32)

# Calculate weight for the positive class
pos = labels.sum().item()
neg = len(labels) - pos
pos_weight = neg / pos

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(
    train_features.numpy(),  # Convert to NumPy for SMOTE
    labels.numpy(),
    test_size=0.2,
    random_state=42,
    stratify=labels.numpy()  # Stratified sampling
)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Convert back to tensors
X_resampled = torch.tensor(X_resampled, dtype=torch.float32)
y_resampled = torch.tensor(y_resampled, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Set up DataLoaders
train_dataset = torch.utils.data.TensorDataset(X_resampled, y_resampled)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Improved model architecture
class RiskClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = RiskClassifier(input_size=train_features.shape[1])
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

# Early stopping parameters
best_auc = 0
patience = 50
no_improve = 0

# Training loop with early stopping
for epoch in range(300):
    model.train()
    epoch_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        epoch_loss += loss.item()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test).squeeze(1)
        val_auc = roc_auc_score(y_test.numpy(), val_outputs.numpy())
        scheduler.step(val_auc)  # Update learning rate

    # Early stopping check
    if val_auc > best_auc:
        best_auc = val_auc
        no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f'Epoch [{epoch+1}] | Loss: {epoch_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')

# Final evaluation
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
with torch.no_grad():
    test_probs = torch.sigmoid(model(X_test)).squeeze()
    test_preds = (test_probs > 0.5).float()

    print(f"\nFinal Performance:")
    print(f"AUC-ROC: {roc_auc_score(y_test, test_probs):.4f}")
    print(f"Accuracy: {accuracy_score(y_test, test_preds):.4f}")
    print(f"F1-Score: {f1_score(y_test, test_preds):.4f}")

# Generate predictions and submission file
submit_df = pd.read_csv('sample_submission.csv')
with torch.no_grad():
    test_outputs = model(test_features)
    test_probs = torch.sigmoid(test_outputs)  # Convert logits to probabilities

submit_df['AccidentFlag'] = test_probs.numpy()  # Probabilities between 0 and 1
submit_df.to_csv('submit.csv', index=False)
