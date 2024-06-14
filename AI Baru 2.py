import pandas as pd
import random
import math

# Mengambil dataset dari file Excel
def load_data(file_path):
    df = pd.read_excel(file_path)
    return df.values.tolist()

# Membersihkan dan memproses dataset
def preprocess_data(data):
    cleaned_dataset = []
    incomplete_data = []
    for index, row in enumerate(data):
        try:
            # Mengonversi kolom ke float
            row[4] = float(row[4])  
            row[5] = float(str(row[5]).replace(',', ''))  
            row[6] = float(str(row[7]).replace(',', '')) 
            
            # Memastikan label valid
            label = str(row[8]).strip().lower()
            if label in ['tinggi', 'rendah']:
                row[8] = label
                cleaned_dataset.append(row)
            else:
                incomplete_data.append(row)
        except ValueError:
            incomplete_data.append(row)
    return cleaned_dataset, incomplete_data

# Mengonversi label ke nilai biner untuk classifier
def encode_labels(dataset):
    label_map = {'tinggi': 1, 'rendah': 0}
    for data in dataset:
        data[8] = label_map[data[8]]
    return dataset

# Membagi dataset menjadi set pelatihan dan pengujian
def train_test_split(X, y, test_size=0.3, random_state=42):
    combined = list(zip(X, y))
    random.seed(random_state)
    random.shuffle(combined)
    split_idx = int(len(combined) * (1 - test_size))
    X_train, y_train = zip(*combined[:split_idx])
    X_test, y_test = zip(*combined[split_idx:])
    return list(X_train), list(X_test), list(y_train), list(y_test)

# Menghitung jumlah kemunculan elemen dalam sebuah list
def count_elements(elements):
    counts = {}
    for element in elements:
        if element in counts:
            counts[element] += 1
        else:
            counts[element] = 1
    return counts

# Menghitung entropi
def calculate_entropy(labels):
    total = len(labels)
    label_counts = count_elements(labels)
    entropy = 0
    for count in label_counts.values():
        probability = count / total
        entropy -= probability * (probability and math.log2(probability))
    return entropy

# Membagi dataset berdasarkan fitur dan threshold
def split_dataset(X, y, feature_index, threshold):
    left_X, left_y, right_X, right_y = [], [], [], []
    for features, label in zip(X, y):
        if features[feature_index] < threshold:
            left_X.append(features)
            left_y.append(label)
        else:
            right_X.append(features)
            right_y.append(label)
    return left_X, left_y, right_X, right_y

# Mencari fitur dan threshold terbaik untuk membagi dataset
def find_best_split(X, y):
    best_gain = 0
    best_feature_index = None
    best_threshold = None
    base_entropy = calculate_entropy(y)
    for feature_index in range(len(X[0])):
        thresholds = set(features[feature_index] for features in X)
        for threshold in thresholds:
            left_X, left_y, right_X, right_y = split_dataset(X, y, feature_index, threshold)
            if len(left_y) == 0 or len(right_y) == 0:
                continue
            left_weight = len(left_y) / len(y)
            right_weight = len(right_y) / len(y)
            new_entropy = left_weight * calculate_entropy(left_y) + right_weight * calculate_entropy(right_y)
            info_gain = base_entropy - new_entropy
            if info_gain > best_gain:
                best_gain = info_gain
                best_feature_index = feature_index
                best_threshold = threshold
    return best_feature_index, best_threshold

# Membangun decision tree secara rekursif
def build_tree(X, y, depth=0, max_depth=None):
    if len(set(y)) == 1:
        return y[0]
    if max_depth is not None and depth >= max_depth:
        return max(set(y), key=y.count)
    
    feature_index, threshold = find_best_split(X, y)
    if feature_index is None:
        return max(set(y), key(y.count))
    
    left_X, left_y, right_X, right_y = split_dataset(X, y, feature_index, threshold)
    left_branch = build_tree(left_X, left_y, depth + 1, max_depth)
    right_branch = build_tree(right_X, right_y, depth + 1, max_depth)
    return (feature_index, threshold, left_branch, right_branch)

# Memprediksi label untuk satu data
def predict_tree(instance, tree):
    if not isinstance(tree, tuple):
        return tree
    feature_index, threshold, left_branch, right_branch = tree
    if instance[feature_index] < threshold:
        return predict_tree(instance, left_branch)
    else:
        return predict_tree(instance, right_branch)

# Mengevaluasi akurasi decision tree
def evaluate_tree(X, y, tree):
    predictions = [predict_tree(instance, tree) for instance in X]
    correct = sum(1 for pred, true in zip(predictions, y) if pred == true)
    accuracy = correct / len(y)
    return accuracy, predictions

# Menghitung confusion matrix
def calculate_confusion_matrix(y_true, y_pred):
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    tn = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
    return [[tn, fp], [fn, tp]]

# Menghitung precision, recall, and F1 score
def calculate_precision_recall_f1(y_true, y_pred):
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    tn = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
    
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
    
    return precision, recall, f1

# Proses utama
file_path = 'C:/Users/LENOVO/Documents/AI/data.xlsx'  
dataset = load_data(file_path)

dataset, incomplete_data = preprocess_data(dataset)
print(f"Total rows after preprocessing: {len(dataset)}")
print(f"Total rows with missing labels: {len(incomplete_data)}")

dataset = encode_labels(dataset)
cleaned_dataset = [[data[4], data[5], data[7], data[8]] for data in dataset if data[8] in [0, 1]]

X = [data[:3] for data in cleaned_dataset]
y = [data[3] for data in cleaned_dataset]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Memeriksa pembagian dataset
print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

# Batasi kedalaman untuk mencegah overfitting
tree = build_tree(X_train, y_train, max_depth=5)  
print("Decision tree built successfully.")

accuracy, predictions = evaluate_tree(X_test, y_test, tree)
print(f"Accuracy: {accuracy}")

# Menghitung dan mengeluarkan hasil confusion matrix
conf_matrix = calculate_confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Menghitung dan mengeluarkan hasil precision, recall, and F1 score
precision, recall, f1 = calculate_precision_recall_f1(y_test, predictions)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Membandingkan dengan label asli untuk mengidentifikasi perbedaan
discrepancies = [(actual, predicted) for actual, predicted in zip(y_test, predictions) if actual != predicted]
print(f"Discrepancies: {discrepancies}")

# Memprediksi label untuk data yang tidak lengkap
for data in incomplete_data:
    features = [data[4], data[5], data[7]]
    predicted_label = predict_tree(features, tree)
    data[8] = 'Tinggi' if predicted_label == 1 else 'Rendah'

# Menulis hasil ke file
output_file_path = 'C:/Users/LENOVO/Documents/AI/manual_predicted_data.xlsx'
results = pd.DataFrame({
    'Name': [data[1] for data in incomplete_data],
    'Loss Rate': [data[4] for data in incomplete_data],
    'Quantity Sold': [data[5] for data in incomplete_data],
    'Total sales in 3 years': [data[7] for data in incomplete_data],
    'Keuntungan': [data[8] for data in incomplete_data]
})
results.to_excel(output_file_path, index=False)
print(f"Results have been written to {output_file_path}")

# Fungsi untuk memprediksi data baru
def predict_new_data(tree):
    while True:
        try:
            loss_rate = float(input("Enter Loss Rate: "))
            quantity_sold = float(input("Enter Quantity Sold: ").replace(',', ''))
            prize_in_3_years = float(input("Enter Prize in 3 years: ").replace(',', ''))
            instance = [loss_rate, quantity_sold, prize_in_3_years]
            prediction = predict_tree(instance, tree)
            print("Prediction:", 'Tingi' if prediction == 1 else 'Rendah')
        except ValueError:
            print("Invalid input, please enter numerical values for Loss Rate, Quantity Sold, and Prize in 3 years.")

# Fungsi menu utama
def main_menu(tree):
    while True:
        print("\nMain Menu:")
        print("1. Predict new data")
        print("2. Exit")
        choice = input("Enter your choice: ")
        if choice == '1':
            predict_new_data(tree)
        elif choice == '2':
            print("Exiting program.")
            break
        else:
            print("Invalid choice, please try again.")

main_menu(tree)
