from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
class DiseaseSymptomSystem:
    def __init__(self, disease_model=None, autoencoder=None, encoder=None, decoder=None):
        self.csv_path = "final_health_dataset.csv"
        self.df = None
        self.X = None
        self.y = None
        self.y_encoded = None
        self.label_encoder = None
        self.num_diseases = None
        self.num_symptoms = 104

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.disease_model = disease_model
        self.autoencoder = autoencoder

        self.encoder = encoder
        self.decoder = decoder

        self.symptom_index_to_name = {}
        self.symptom_name_to_index = {}

    def load_data(self):
        self.df = pd.read_csv(self.csv_path)
        self.X = self.df.iloc[:, 1:].values.astype(np.float32)
        self.y = self.df.iloc[:, 0].values

        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        self.num_diseases = len(np.unique(self.y_encoded))

        self._build_symptom_mappings()

    def _build_symptom_mappings(self):
        symptom_columns = self.df.columns[1:]
        for idx, symptom_name in enumerate(symptom_columns):
            self.symptom_index_to_name[idx] = symptom_name
            self.symptom_name_to_index[symptom_name] = idx

    def preprocess_data(self, test_size=0.2, random_state=42):
        y_one_hot = keras.utils.to_categorical(self.y_encoded, num_classes=self.num_diseases)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, y_one_hot, test_size=test_size, stratify=self.y_encoded, random_state=random_state
        )

    def build_disease_model(self):
        inputs = layers.Input(shape=(self.num_symptoms,))

        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        outputs = layers.Dense(self.num_diseases, activation='softmax')(x)

        self.disease_model = keras.Model(inputs=inputs, outputs=outputs)
        self.disease_model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )

    def build_autoencoder(self):
        inputs = layers.Input(shape=(self.num_symptoms,))

        encoded = layers.Dense(64, activation='relu')(inputs)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(32, activation='relu')(encoded)

        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(self.num_symptoms, activation='sigmoid')(decoded)

        self.autoencoder = keras.Model(inputs=inputs, outputs=decoded)
        self.autoencoder.compile(
            loss='binary_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=['mse']
        )

        self.encoder = keras.Model(inputs=inputs, outputs=encoded)

    def train_disease_model(self, epochs=50, batch_size=32, validation_split=0.2):
        train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((self.X_test, self.y_test))
        val_dataset = val_dataset.batch(batch_size)

        self.disease_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            verbose=1
        )

        test_loss, test_accuracy = self.disease_model.evaluate(val_dataset, verbose=0)
        print(f"Kardick Accuracy Gawk: {test_accuracy:.4f}")

    def train_autoencoder(self, epochs=50, batch_size=32):
        train_dataset = tf.data.Dataset.from_tensor_slices((self.X, self.X))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        self.autoencoder.fit(
            train_dataset,
            epochs=epochs,
            verbose=1
        )

    def predict_disease(self, symptom_vector):
        symptom_array = np.array(symptom_vector, dtype=np.float32).reshape(1, -1)
        prediction = self.disease_model.predict(symptom_array, verbose=0)

        disease_idx = np.argmax(prediction[0])
        disease_label = self.label_encoder.inverse_transform([disease_idx])[0]
        probabilities = prediction[0]

        return disease_label, probabilities

    def recommend_symptoms(self, selected_symptoms, top_k=5):
        if isinstance(selected_symptoms, list) and all(isinstance(x, int) for x in selected_symptoms):
            symptom_vector = np.zeros(self.num_symptoms, dtype=np.float32)
            for idx in selected_symptoms:
                if 0 <= idx < self.num_symptoms:
                    symptom_vector[idx] = 1.0
        else:
            symptom_vector = np.array(selected_symptoms, dtype=np.float32)

        symptom_array = symptom_vector.reshape(1, -1)
        predictions = self.autoencoder.predict(symptom_array, verbose=0)[0]

        predictions[symptom_vector > 0] = 0.0

        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_scores = predictions[top_indices]

        return list(zip(top_indices, top_scores))

    def predict_disease_from_names(self, symptom_names_list):
        symptom_vector = np.zeros(self.num_symptoms, dtype=np.float32)

        for name in symptom_names_list:
            if name in self.symptom_name_to_index:
                idx = self.symptom_name_to_index[name]
                symptom_vector[idx] = 1.0

        return self.predict_disease(symptom_vector)

    def recommend_symptoms_from_names(self, symptom_names_list, top_k=5):
        selected_indices = []
        for name in symptom_names_list:
            if name in self.symptom_name_to_index:
                selected_indices.append(self.symptom_name_to_index[name])

        recommendations = self.recommend_symptoms(selected_indices, top_k=top_k)

        result = []
        for symptom_idx, score in recommendations:
            symptom_name = self.symptom_index_to_name[symptom_idx]
            result.append((symptom_name, float(score)))

        return result
disease_model=tf.keras.models.load_model("weights_disease/disease_model")
autoencoder=tf.keras.models.load_model("weights_disease/autoencoder/autoencoder")
encoder=tf.keras.models.load_model("weights_disease/encoder/encoder")
system = DiseaseSymptomSystem(disease_model, autoencoder, encoder)
print("Loading data...")
system.load_data()
print("Preprocessing data...")
system.preprocess_data(test_size=0.2, random_state=42)


symptom_names_input = ['depression', 'depressive or psychotic symptoms'] #<==== Symptoms ikkada like once manam 5 or any symptoms teskunte we will predict the upcoming symptoms to make the user work easy as a doc

print (symptom_names_input)
disease_label, probabilities = system.predict_disease_from_names(symptom_names_input)


print(f"Predicted Disease: {disease_label}")
top_3_indices = np.argsort(probabilities)[-3:][::-1]
for idx in top_3_indices:
    disease_name = system.label_encoder.inverse_transform([idx])[0]
    prob = probabilities[idx]
    print(f"  {disease_name}: {prob:.4f}")


    
print("\nRunning symptom recommendation...")
recommended = system.recommend_symptoms_from_names(symptom_names_input, top_k=5)
print(f"Recommended Symptoms (given input symptoms):")

for symptom_name, score in recommended:
    print(f"  {symptom_name}: {score:.4f}")
