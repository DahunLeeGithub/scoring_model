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
    # period should be "T-2_" for two periods ago or "T-1_" for last period
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
        "T-2_CashAndDeposits,T-2_NotesReceivable,T-2_AccountsReceivable,T-2_OtherCurrentAssets,"
        "T-2_CompletedProjectsAccruedIncome,T-2_ConstructionInProgressExpenses,T-2_Inventory,"
        "T-2_OtherCurrentAssets2,T-2_ShortTermLoansReceivable,T-2_CurrentAssets,"
        "T-2_AllowanceForDoubtfulAccounts,T-2_TangibleFixedAssets,T-2_IntangibleFixedAssets,"
        "T-2_InvestmentsAndOtherAssets,T-2_DeferredAssets,T-2_LongTermLoansReceivable,"
        "T-2_FixedAssets,T-2_TotalAssets,T-2_NotesPayable,T-2_AccountsPayable,T-2_ShortTermBorrowings,"
        "T-2_AccruedExpenses,T-2_Provisions,T-2_AdvanceReceivedForConstruction,T-2_AdvanceReceived,"
        "T-2_OtherCurrentLiabilities,T-2_CurrentLiabilities,T-2_LongTermBorrowings,"
        "T-2_OtherNonCurrentLiabilities,T-2_NonCurrentLiabilities,T-2_TotalLiabilities,"
        "T-2_CapitalStock,T-2_Reserves,T-2_UndistributedProfitBS,T-2_TotalEquity,"
        "T-2_TotalLiabilitiesEquity,T-2_AssetLiabilityError,T-2_DiscountedNotes,T-2_EndorsedNotes,"
        "T-2_Sales,T-2_CostOfGoodsSold,T-2_GrossProfit,T-2_SG&AExpenses,T-2_OperatingIncome,"
        "T-2_InterestIncome,T-2_OtherNonOperatingIncome,T-2_TotalNonOperatingIncome,"
        "T-2_InterestExpense,T-2_OtherNonOperatingExpenses,T-2_TotalNonOperatingExpenses,"
        "T-2_OrdinaryIncome,T-2_SpecialGains,T-2_SpecialLosses,T-2_IncomeBeforeTax,T-2_CorporateTax,"
        "T-2_NetIncome,T-2_CarryForwardProfit,T-2_InterimDividends,T-2_UndistributedProfitPL,"
        "T-2_BSPL_Error,T-2_CarryForwardProfitError,T-2_SpecialDepreciationReserve,T-2_Depreciation,"
        "T-1_CashAndDeposits,T-1_NotesReceivable,T-1_AccountsReceivable,T-1_OtherCurrentAssets,"
        "T-1_CompletedProjectsAccruedIncome,T-1_ConstructionInProgressExpenses,T-1_Inventory,"
        "T-1_OtherCurrentAssets2,T-1_ShortTermLoansReceivable,T-1_CurrentAssets,"
        "T-1_AllowanceForDoubtfulAccounts,T-1_TangibleFixedAssets,T-1_IntangibleFixedAssets,"
        "T-1_InvestmentsAndOtherAssets,T-1_DeferredAssets,T-1_LongTermLoansReceivable,"
        "T-1_FixedAssets,T-1_TotalAssets,T-1_NotesPayable,T-1_AccountsPayable,T-1_ShortTermBorrowings,"
        "T-1_AccruedExpenses,T-1_Provisions,T-1_AdvanceReceivedForConstruction,T-1_AdvanceReceived,"
        "T-1_OtherCurrentLiabilities,T-1_CurrentLiabilities,T-1_LongTermBorrowings,"
        "T-1_OtherNonCurrentLiabilities,T-1_NonCurrentLiabilities,T-1_TotalLiabilities,"
        "T-1_CapitalStock,T-1_Reserves,T-1_UndistributedProfitBS,T-1_TotalEquity,"
        "T-1_TotalLiabilitiesEquity,T-1_AssetLiabilityError,T-1_DiscountedNotes,T-1_EndorsedNotes,"
        "T-1_Sales,T-1_CostOfGoodsSold,T-1_GrossProfit,T-1_SG&AExpenses,T-1_OperatingIncome,"
        "T-1_InterestIncome,T-1_OtherNonOperatingIncome,T-1_TotalNonOperatingIncome,"
        "T-1_InterestExpense,T-1_OtherNonOperatingExpenses,T-1_TotalNonOperatingExpenses,"
        "T-1_OrdinaryIncome,T-1_SpecialGains,T-1_SpecialLosses,T-1_IncomeBeforeTax,T-1_CorporateTax,"
        "T-1_NetIncome,T-1_CarryForwardProfit,T-1_InterimDividends,T-1_UndistributedProfitPL,"
        "T-1_BSPL_Error,T-1_CarryForwardProfitError,T-1_SpecialDepreciationReserve,T-1_Depreciation,"
        "T-2_FiscalPeriod,T-1_FiscalPeriod"
    )
    cols = col_text.split(',')
    # Remove columns that are not needed
    cols_remove = ['FiscalPeriod']
    cols_remove = [f'T-2_{col}' for col in cols_remove] + [f'T-1_{col}' for col in cols_remove]
    cols = [col for col in cols if col not in cols_remove]

    data = origin_data[cols].copy()
    data = data.replace(',', '', regex=True)
    data = data.apply(pd.to_numeric, errors='coerce')

    # One-hot encode the industry code
    dummies = pd.get_dummies(data['IndustryCode'], prefix='IndustryCode', drop_first=True)
    data = pd.concat([data.drop('IndustryCode', axis=1), dummies], axis=1)

    # Calculate various financial ratios
    data['T-2_CurrentRatio'] = data['T-2_CurrentAssets'] / data['T-2_CurrentLiabilities'].replace(0, EPSILON)
    data['T-1_CurrentRatio'] = data['T-1_CurrentAssets'] / data['T-1_CurrentLiabilities'].replace(0, EPSILON)

    data['T-2_QuickRatio'] = (data['T-2_CurrentAssets'] - data['T-2_Inventory']) / data['T-2_CurrentLiabilities'].replace(0, EPSILON)
    data['T-1_QuickRatio'] = (data['T-1_CurrentAssets'] - data['T-1_Inventory']) / data['T-1_CurrentLiabilities'].replace(0, EPSILON)

    data['T-2_CashRatio'] = data['T-2_CashAndDeposits'] / data['T-2_CurrentAssets'].replace(0, EPSILON)
    data['T-1_CashRatio'] = data['T-1_CashAndDeposits'] / data['T-1_CurrentAssets'].replace(0, EPSILON)

    data['T-2_DebtRatio'] = data['T-2_TotalLiabilities'] / data['T-2_TotalAssets'].replace(0, EPSILON)
    data['T-1_DebtRatio'] = data['T-1_TotalLiabilities'] / data['T-1_TotalAssets'].replace(0, EPSILON)

    data['T-2_InterestCoverageRatio'] = data['T-2_OperatingIncome'] / data['T-2_InterestExpense'].replace(0, EPSILON)
    data['T-1_InterestCoverageRatio'] = data['T-1_OperatingIncome'] / data['T-1_InterestExpense'].replace(0, EPSILON)

    data['T-2_EquityRatio'] = data['T-2_TotalLiabilities'] / data['T-2_TotalEquity'].replace(0, EPSILON)
    data['T-1_EquityRatio'] = data['T-1_TotalLiabilities'] / data['T-1_TotalEquity'].replace(0, EPSILON)

    data['T-2_GrossMargin'] = data['T-2_GrossProfit'] / data['T-2_Sales'].replace(0, EPSILON)
    data['T-1_GrossMargin'] = data['T-1_GrossProfit'] / data['T-1_Sales'].replace(0, EPSILON)

    data['T-2_OperatingMargin'] = data['T-2_OperatingIncome'] / data['T-2_Sales'].replace(0, EPSILON)
    data['T-1_OperatingMargin'] = data['T-1_OperatingIncome'] / data['T-1_Sales'].replace(0, EPSILON)

    data['T-2_NetMargin'] = data['T-2_NetIncome'] / data['T-2_Sales'].replace(0, EPSILON)
    data['T-1_NetMargin'] = data['T-1_NetIncome'] / data['T-1_Sales'].replace(0, EPSILON)

    data['T-2_TotalAssetTurnover'] = data['T-2_Sales'] / data['T-2_TotalAssets'].replace(0, EPSILON)
    data['T-1_TotalAssetTurnover'] = data['T-1_Sales'] / data['T-1_TotalAssets'].replace(0, EPSILON)

    data['T-2_ReceivablesTurnover'] = data['T-2_Sales'] / data['T-2_AccountsReceivable'].replace(0, EPSILON)
    data['T-1_ReceivablesTurnover'] = data['T-1_Sales'] / data['T-1_AccountsReceivable'].replace(0, EPSILON)

    data['T-2_InventoryTurnover'] = data['T-2_Sales'] / data['T-2_Inventory'].replace(0, EPSILON)
    data['T-1_InventoryTurnover'] = data['T-1_Sales'] / data['T-1_Inventory'].replace(0, EPSILON)

    data['ProfitVolatility'] = abs(data['T-1_OperatingIncome'] - data['T-2_OperatingIncome']) / data['T-2_OperatingIncome'].replace(0, EPSILON)

    data['T-2_NegativeCashFlow'] = (data['T-2_OperatingIncome'] < 0) | (data['T-2_CashAndDeposits'] < data['T-2_ShortTermBorrowings'])
    data['T-1_NegativeCashFlow'] = (data['T-1_OperatingIncome'] < 0) | (data['T-1_CashAndDeposits'] < data['T-1_ShortTermBorrowings'])

    data['T-1_CumulativeLoss'] = data['T-1_UndistributedProfitBS'] < 0

    data['SalesGrowthRate'] = (data['T-1_Sales'] - data['T-2_Sales']) / data['T-2_Sales'].replace(0, EPSILON)
    data['COGSGrowthRate'] = (data['T-1_CostOfGoodsSold'] - data['T-2_CostOfGoodsSold']) / data['T-2_CostOfGoodsSold'].replace(0, EPSILON)
    data['OperatingIncomeGrowthRate'] = (data['T-1_OperatingIncome'] - data['T-2_OperatingIncome']) / data['T-2_OperatingIncome'].replace(0, EPSILON)
    data['DebtGrowthRate'] = (data['T-1_TotalLiabilities'] - data['T-2_TotalLiabilities']) / data['T-2_TotalLiabilities'].replace(0, EPSILON)

    data['T-2_Z-Score'] = calculate_altman_z_score(data, 'T-2_')
    data['T-1_Z-Score'] = calculate_altman_z_score(data, 'T-1_')

    # Specify columns for which change rates are needed (excluding some columns)
    before_cols = [col for col in cols if col.startswith('T-1_')]
    no_change_cols = ['AssetGrowthRate', 'FSRI', 'T-1_CumulativeLoss', 'SalesGrowthRate', 'COGSGrowthRate', 'OperatingIncomeGrowthRate', 'DebtGrowthRate']
    for col in before_cols:
        if col in no_change_cols:
            continue
        data[col[4:] + '_ChangeRate'] = (data[col] - data[col.replace('T-1_', 'T-2_', 1)]) / data[col.replace('T-1_', 'T-2_', 1)]

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
