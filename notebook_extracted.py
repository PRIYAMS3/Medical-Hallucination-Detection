# ===== Cell 0 =====
# Install required libraries

!pip install shap scikit-learn



import torch

import torch.nn as nn

import torch.optim as optim



import pandas as pd

import numpy as np



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.utils.class_weight import compute_class_weight



from torch.utils.data import Dataset, DataLoader



import random

# ===== Cell 1 =====
def set_seed(seed=42):

    torch.manual_seed(seed)

    np.random.seed(seed)

    random.seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)



set_seed(42)

# ===== Cell 2 =====
df = pd.read_csv("output.csv")



print(df.shape)

df.head()

# ===== Cell 3 =====
print(df['Result'].value_counts())

# ===== Cell 4 =====
label_mapping = {-1: 0, 1: 1}

df['Result'] = df['Result'].map(label_mapping)

# ===== Cell 5 =====
nn.Linear(64, 2)

# ===== Cell 6 =====
nn.CrossEntropyLoss()

# ===== Cell 7 =====
target_col = 'Result'



label_mapping = {-1: 0, 1: 1}

df[target_col] = df[target_col].map(label_mapping)



X = df.drop(columns=[target_col])

y = df[target_col]



print(y.value_counts())

# ===== Cell 8 =====
print(df.isnull().sum().sum())



df = df.fillna(0)

# ===== Cell 9 =====
print(df.isnull().sum())

# ===== Cell 10 =====
# Target column

target_col = 'Result'



# Convert labels (Binary)

label_mapping = {-1: 0, 1: 1}

df[target_col] = df[target_col].map(label_mapping)



# Handle missing values (safe for this dataset)

df = df.fillna(0)



# Split features & target

X = df.drop(columns=[target_col])

y = df[target_col]



print("Target distribution:\n", y.value_counts())

print("Missing values:", df.isnull().sum().sum())

# ===== Cell 11 =====
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

    X.values, y,

    test_size=0.2,

    stratify=y,

    random_state=42

)



print(X_train.shape, X_test.shape)

# ===== Cell 12 =====
from sklearn.utils.class_weight import compute_class_weight

import torch

import numpy as np



class_weights = compute_class_weight(

    class_weight='balanced',

    classes=np.unique(y_train),

    y=y_train

)



class_weights = torch.tensor(class_weights, dtype=torch.float)

print("Class Weights:", class_weights)

# ===== Cell 13 =====
from torch.utils.data import Dataset, DataLoader



class PhishingDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)

        self.y = torch.tensor(y.values, dtype=torch.long)



    def __len__(self):

        return len(self.y)



    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]





train_dataset = PhishingDataset(X_train, y_train)

test_dataset = PhishingDataset(X_test, y_test)



train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ===== Cell 14 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# ===== Cell 15 =====
import torch.nn as nn



class SimpleANN(nn.Module):

    def __init__(self, input_dim):

        super(SimpleANN, self).__init__()



        self.model = nn.Sequential(

            nn.Linear(input_dim, 64),

            nn.ReLU(),

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Linear(32, 2)   # 2 classes (binary)

        )



    def forward(self, x):

        return self.model(x)

# ===== Cell 16 =====
import torch.optim as optim



def train_model(model, train_loader, epochs=20):

    model.to(device)



    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = optim.Adam(model.parameters(), lr=0.001)



    for epoch in range(epochs):

        model.train()

        total_loss = 0



        for X_batch, y_batch in train_loader:

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)



            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)



            loss.backward()

            optimizer.step()



            total_loss += loss.item()



        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")



    return model

# ===== Cell 17 =====
from sklearn.metrics import classification_report, confusion_matrix



def evaluate_model(model, test_loader):

    model.eval()



    all_preds = []

    all_labels = []



    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(device)



            outputs = model(X_batch)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()



            all_preds.extend(preds)

            all_labels.extend(y_batch.numpy())



    print("Classification Report:\n")

    print(classification_report(all_labels, all_preds))



    print("Confusion Matrix:\n")

    print(confusion_matrix(all_labels, all_preds))

# ===== Cell 18 =====
input_dim = X_train.shape[1]



model = SimpleANN(input_dim)



model = train_model(model, train_loader, epochs=20)



evaluate_model(model, test_loader)

# ===== Cell 19 =====
class DeepANN(nn.Module):

    def __init__(self, input_dim):

        super(DeepANN, self).__init__()



        self.model = nn.Sequential(

            nn.Linear(input_dim, 128),

            nn.BatchNorm1d(128),

            nn.ReLU(),



            nn.Linear(128, 64),

            nn.BatchNorm1d(64),

            nn.ReLU(),



            nn.Linear(64, 32),

            nn.ReLU(),



            nn.Linear(32, 2)

        )



    def forward(self, x):

        return self.model(x)

# ===== Cell 20 =====
model2 = DeepANN(input_dim)



model2 = train_model(model2, train_loader, epochs=20)



evaluate_model(model2, test_loader)

# ===== Cell 21 =====
class DropoutANN(nn.Module):

    def __init__(self, input_dim):

        super(DropoutANN, self).__init__()



        self.model = nn.Sequential(

            nn.Linear(input_dim, 128),

            nn.ReLU(),

            nn.Dropout(0.3),



            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Dropout(0.3),



            nn.Linear(64, 32),

            nn.ReLU(),



            nn.Linear(32, 2)

        )



    def forward(self, x):

        return self.model(x)

# ===== Cell 22 =====
model3 = DropoutANN(input_dim)



model3 = train_model(model3, train_loader, epochs=20)



evaluate_model(model3, test_loader)

# ===== Cell 23 =====
class ResidualBlock(nn.Module):

    def __init__(self, dim):

        super(ResidualBlock, self).__init__()



        self.block = nn.Sequential(

            nn.Linear(dim, dim),

            nn.ReLU(),

            nn.Linear(dim, dim)

        )



    def forward(self, x):

        return x + self.block(x)





class ResidualMLP(nn.Module):

    def __init__(self, input_dim):

        super(ResidualMLP, self).__init__()



        self.input_layer = nn.Linear(input_dim, 64)



        self.res1 = ResidualBlock(64)

        self.res2 = ResidualBlock(64)



        self.output_layer = nn.Linear(64, 2)



    def forward(self, x):

        x = torch.relu(self.input_layer(x))

        x = self.res1(x)

        x = self.res2(x)

        return self.output_layer(x)

# ===== Cell 24 =====
model4 = ResidualMLP(input_dim)



model4 = train_model(model4, train_loader, epochs=20)



evaluate_model(model4, test_loader)

# ===== Cell 25 =====
class WideDeep(nn.Module):

    def __init__(self, input_dim):

        super(WideDeep, self).__init__()



        # Deep part

        self.deep = nn.Sequential(

            nn.Linear(input_dim, 64),

            nn.ReLU(),

            nn.Linear(64, 32),

            nn.ReLU()

        )



        # Output combines raw + deep

        self.output = nn.Linear(input_dim + 32, 2)



    def forward(self, x):

        deep_out = self.deep(x)

        combined = torch.cat([x, deep_out], dim=1)

        return self.output(combined)

# ===== Cell 26 =====
model5 = WideDeep(input_dim)



model5 = train_model(model5, train_loader, epochs=20)



evaluate_model(model5, test_loader)

# ===== Cell 27 =====
class CNN1D(nn.Module):

    def __init__(self, input_dim):

        super(CNN1D, self).__init__()



        self.conv = nn.Sequential(

            nn.Conv1d(1, 16, kernel_size=3, padding=1),

            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),

            nn.ReLU()

        )



        self.fc = nn.Sequential(

            nn.Linear(32 * input_dim, 64),

            nn.ReLU(),

            nn.Linear(64, 2)

        )



    def forward(self, x):

        x = x.unsqueeze(1)  # (batch, 1, features)

        x = self.conv(x)

        x = x.view(x.size(0), -1)

        return self.fc(x)

# ===== Cell 28 =====
model6 = CNN1D(input_dim)



model6 = train_model(model6, train_loader, epochs=20)



evaluate_model(model6, test_loader)

# ===== Cell 29 =====
class AttentionModel(nn.Module):

    def __init__(self, input_dim):

        super(AttentionModel, self).__init__()



        self.attention = nn.Sequential(

            nn.Linear(input_dim, input_dim),

            nn.Softmax(dim=1)

        )



        self.fc = nn.Sequential(

            nn.Linear(input_dim, 64),

            nn.ReLU(),

            nn.Linear(64, 2)

        )



    def forward(self, x):

        attn_weights = self.attention(x)

        x = x * attn_weights

        return self.fc(x)

# ===== Cell 30 =====
model7 = AttentionModel(input_dim)



model7 = train_model(model7, train_loader, epochs=20)



evaluate_model(model7, test_loader)

# ===== Cell 31 =====
def ensemble_predict(models, test_loader):

    for m in models:

        m.eval()



    all_preds = []

    all_labels = []



    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(device)



            probs = []



            for m in models:

                outputs = m(X_batch)

                prob = torch.softmax(outputs, dim=1)

                probs.append(prob)



            avg_prob = torch.mean(torch.stack(probs), dim=0)

            preds = torch.argmax(avg_prob, dim=1).cpu().numpy()



            all_preds.extend(preds)

            all_labels.extend(y_batch.numpy())



    from sklearn.metrics import classification_report, confusion_matrix



    print("Ensemble Results:\n")

    print(classification_report(all_labels, all_preds))

    print(confusion_matrix(all_labels, all_preds))

# ===== Cell 32 =====
ensemble_models = [model, model2, model4]  # simple, deep, residual



ensemble_predict(ensemble_models, test_loader)

# ===== Cell 33 =====
import pandas as pd



results = [

    ["Simple ANN", 0.96, 0.96, 0.96],

    ["Deep ANN (BatchNorm)", 0.97, 0.97, 0.97],

    ["Dropout ANN", 0.97, 0.97, 0.97],

    ["Residual MLP", 0.97, 0.97, 0.97],

    ["Wide & Deep", 0.96, 0.96, 0.96],

    ["CNN", 0.96, 0.96, 0.96],

    ["Attention", 0.95, 0.95, 0.95],

    ["Ensemble", 0.97, 0.97, 0.97]

]



df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "F1-score"])



df_results = df_results.sort_values(by="F1-score", ascending=False)



df_results

# ===== Cell 34 =====
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize



def plot_roc(model, test_loader):

    model.eval()

    y_true = []

    y_scores = []



    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(device)

            outputs = model(X_batch)

            probs = torch.softmax(outputs, dim=1)[:,1]



            y_true.extend(y_batch.numpy())

            y_scores.extend(probs.cpu().numpy())



    fpr, tpr, _ = roc_curve(y_true, y_scores)

    roc_auc = auc(fpr, tpr)



    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")

    plt.plot([0,1],[0,1],'--')

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.show()

# ===== Cell 35 =====
plot_roc(model2, test_loader)   # best ANN

# ===== Cell 36 =====
!pip install shap

# ===== Cell 37 =====
import torch



X_sample = torch.tensor(X_train[:100], dtype=torch.float32).to(device)

# ===== Cell 38 =====
import shap

import numpy as np



# move model to cpu

model2_cpu = model2.to("cpu")



# sample

X_sample = X_train[:100]



# prediction function

def model_predict(x):

    x_tensor = torch.tensor(x, dtype=torch.float32)

    with torch.no_grad():

        outputs = model2_cpu(x_tensor)

        probs = torch.softmax(outputs, dim=1).numpy()

    return probs



# explainer

explainer = shap.KernelExplainer(model_predict, X_sample)



# shap values

shap_values = explainer.shap_values(X_sample[:50])

# ===== Cell 39 =====
print(type(shap_values))

print(len(shap_values))



for i in range(len(shap_values)):

    print(f"Class {i} shape:", np.array(shap_values[i]).shape)

# ===== Cell 40 =====
# Save best model

torch.save(model2.state_dict(), "best_model.pt")



# Save results

df_results.to_csv("results.csv", index=False)

# ===== Cell 41 =====
shap_array = np.array(shap_values)

shap_array = shap_array[:, :, 1]



shap.summary_plot(shap_array, X_sample[:50], feature_names=X.columns)

# ===== Cell 42 =====
model2.load_state_dict(torch.load("best_model.pt"))

model2.to(device)

model2.eval()

# ===== Cell 43 =====
feature_index = {col: i for i, col in enumerate(X.columns)}

# ===== Cell 44 =====
def model_predict_with_confidence(model, x):

    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)



    with torch.no_grad():

        output = model(x_tensor)

        probs = torch.softmax(output, dim=1)



    confidence, pred = torch.max(probs, dim=1)



    return pred.item(), confidence.item()

# ===== Cell 45 =====
def rule_engine(x):

    rules_triggered = []



    if x[feature_index['having_IP_Address']] == 1:

        rules_triggered.append("IP address used")



    if x[feature_index['SSLfinal_State']] == -1:

        rules_triggered.append("Invalid SSL")



    if x[feature_index['URL_of_Anchor']] == 1:

        rules_triggered.append("Suspicious anchor URLs")



    if x[feature_index['Prefix_Suffix']] == 1:

        rules_triggered.append("Hyphen in domain")



    return rules_triggered

# ===== Cell 46 =====
def phishing_detection_system(model, x, threshold=0.8):



    pred, conf = model_predict_with_confidence(model, x)

    rules = rule_engine(x)



    # decision logic

    if conf < threshold and len(rules) > 0:

        final_pred = 1  # phishing (rule override)

        decision_type = "Rule-based override"

    else:

        final_pred = pred

        decision_type = "Model-based"



    label = "Phishing" if final_pred == 1 else "Legitimate"



    return {

        "Prediction": label,

        "Confidence": round(conf, 3),

        "Decision Type": decision_type,

        "Rules Triggered": rules

    }

# ===== Cell 47 =====
sample = X_test[0]



result = phishing_detection_system(model2, sample)



print(result)

# ===== Cell 48 =====
idx = int(input("Enter test index: "))

sample = X_test[idx]



print(phishing_detection_system(model2, sample))

# ===== Cell 49 =====
for i in range(5):

    print(f"\nSample {i}")

    print(phishing_detection_system(model2, X_test[i]))

# ===== Cell 50 =====

