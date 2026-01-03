import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Step 1: Load dataset
data = pd.read_csv("dataset.csv")

# Last column is label
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# ================================================
# Implement BKA-inspired feature selection
# ================================================
def fitness_function(X_fs, y):
    # Using a simple classifier accuracy as fitness
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    scores = cross_val_score(clf, X_fs, y, cv=3, scoring='accuracy')
    return scores.mean()

def black_winged_kite_algorithm(X, y, n_features_to_select=10, max_iter=20):
    n_samples, n_features = X.shape
    
    # Initialize population (set of feature subsets)
    population_size = 10
    population = []
    for _ in range(population_size):
        feature_mask = np.zeros(n_features, dtype=bool)
        selected_idx = np.random.choice(n_features, n_features_to_select, replace=False)
        feature_mask[selected_idx] = True
        population.append(feature_mask)
    
    best_mask = population[0]
    best_fitness = 0
    
    for iteration in range(max_iter):
        new_population = []
        for mask in population:
            # Generate neighbor solutions by flipping one bit
            neighbors = []
            for i in range(n_features):
                new_mask = mask.copy()
                new_mask[i] = not new_mask[i]
                # Make sure exactly n_features_to_select are True
                if new_mask.sum() == n_features_to_select:
                    neighbors.append(new_mask)
            
            # Evaluate neighbors
            fitnesses = [fitness_function(X[:, nb], y) for nb in neighbors]
            max_idx = np.argmax(fitnesses)
            best_neighbor = neighbors[max_idx]
            best_neighbor_fitness = fitnesses[max_idx]
            
            # If neighbor is better, replace
            if best_neighbor_fitness > fitness_function(X[:, mask], y):
                new_population.append(best_neighbor)
                if best_neighbor_fitness > best_fitness:
                    best_fitness = best_neighbor_fitness
                    best_mask = best_neighbor
            else:
                new_population.append(mask)
        
        population = new_population
    
    return X[:, best_mask]

# Select features
X_selected = black_winged_kite_algorithm(X, y, n_features_to_select=min(10, X.shape[1]), max_iter=10)

# Prepare data for GCN-BiLSTM
X_reshaped = X_selected.reshape((X_selected.shape[0], X_selected.shape[1], 1))

# Convert labels to categorical if needed
unique_classes = np.unique(y)
if len(unique_classes) > 2:
    y_cat = to_categorical(y)
else:
    y_cat = y

X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_cat, test_size=0.2, random_state=42)

# GCN-BiLSTM model
inputs = Input(shape=(X_reshaped.shape[1], 1))
x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
x = MaxPooling1D(pool_size=2)(x)
x = Bidirectional(LSTM(50, return_sequences=False))(x)
x = Flatten()(x)
x = Dense(100, activation='relu')(x)

if len(unique_classes) > 2:
    outputs = Dense(len(unique_classes), activation='softmax')(x)
    loss = 'categorical_crossentropy'
else:
    outputs = Dense(1, activation='sigmoid')(x)
    loss = 'binary_crossentropy'

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
X_train_feats = feature_extractor.predict(X_train)
X_test_feats = feature_extractor.predict(X_test)

# Random Forest
if len(unique_classes) > 2:
    y_train_rf = np.argmax(y_train, axis=1)
    y_test_rf = np.argmax(y_test, axis=1)
else:
    y_train_rf = y_train
    y_test_rf = y_test

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_feats, y_train_rf)
y_pred = rf.predict(X_test_feats)

print(classification_report(y_test_rf, y_pred))
