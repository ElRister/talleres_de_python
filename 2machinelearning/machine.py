import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFormLayout, QLineEdit, QMessageBox
from sklearn.ensemble import RandomForestClassifier

class LungCancerPredictor(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()
        self.load_data()
        self.train_model()

    def init_ui(self):
        self.setWindowTitle("Predicción de Cáncer de Pulmón")
        self.setGeometry(100, 100, 400, 250)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.form_layout = QFormLayout()

        self.gender_edit = QLineEdit()
        self.form_layout.addRow("Género (1=M, 2=F):", self.gender_edit)

        self.age_edit = QLineEdit()
        self.form_layout.addRow("Edad:", self.age_edit)

        self.smoking_edit = QLineEdit()
        self.form_layout.addRow("Fumador (2=Si, 1=No):", self.smoking_edit)

        self.yellow_fingers_edit = QLineEdit()
        self.form_layout.addRow("Dedos amarillos (2=Si, 1=No):", self.yellow_fingers_edit)

        self.anxiety_edit = QLineEdit()
        self.form_layout.addRow("Ansiedad (2=Si, 1=No):", self.anxiety_edit)

        self.peer_pressure_edit = QLineEdit()
        self.form_layout.addRow("Presión de grupo (2=Si, 1=No):", self.peer_pressure_edit)

        self.chronic_disease_edit = QLineEdit()
        self.form_layout.addRow("Enfermedad crónica (2=Si, 1=No):", self.chronic_disease_edit)

        self.fatigue_edit = QLineEdit()
        self.form_layout.addRow("Fatiga (2=Si, 1=No):", self.fatigue_edit)

        self.allergy_edit = QLineEdit()
        self.form_layout.addRow("Alergias (2=Si, 1=No):", self.allergy_edit)

        self.wheezing_edit = QLineEdit()
        self.form_layout.addRow("Silbidos en el pecho (2=Si, 1=No):", self.wheezing_edit)

        self.alcohol_consuming_edit = QLineEdit()
        self.form_layout.addRow("Consumo de alcohol (2=Si, 1=No):", self.alcohol_consuming_edit)

        self.coughing_edit = QLineEdit()
        self.form_layout.addRow("Tos (2=Si, 1=No):", self.coughing_edit)

        self.shortness_of_breath_edit = QLineEdit()
        self.form_layout.addRow("Dificultad para respirar (2=Si, 1=No):", self.shortness_of_breath_edit)

        self.swallowing_difficulty_edit = QLineEdit()
        self.form_layout.addRow("Dificultad para tragar (2=Si, 1=No):", self.swallowing_difficulty_edit)

        self.chest_pain_edit = QLineEdit()
        self.form_layout.addRow("Dolor en el pecho (2=Si, 1=No):", self.chest_pain_edit)

        self.layout.addLayout(self.form_layout)

        self.result_label = QLabel("Resultado de la predicción: -")
        self.layout.addWidget(self.result_label)

        self.predict_button = QPushButton("Predecir")
        self.predict_button.clicked.connect(self.predict)
        self.layout.addWidget(self.predict_button)

        self.central_widget.setLayout(self.layout)

    def load_data(self):
        try:
            self.data = pd.read_excel("PACIENTES.xlsx")
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "El archivo 'PACIENTE.xlsx' no se encuentra.")
            sys.exit(1)

    def train_model(self):
        self.X = self.data.drop("LUNG_CANCER", axis=1)
        self.y = self.data["LUNG_CANCER"]

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X, self.y)

    def predict(self):
        try:
            patient_data = [
                float(self.age_edit.text()),
                int(self.gender_edit.text()),
                int(self.smoking_edit.text()),
                int(self.yellow_fingers_edit.text()),
                int(self.anxiety_edit.text()),
                int(self.peer_pressure_edit.text()),
                int(self.chronic_disease_edit.text()),
                int(self.fatigue_edit.text()),
                int(self.allergy_edit.text()),
                int(self.wheezing_edit.text()),
                int(self.alcohol_consuming_edit.text()),
                int(self.coughing_edit.text()),
                int(self.shortness_of_breath_edit.text()),
                int(self.swallowing_difficulty_edit.text()),
                int(self.chest_pain_edit.text()),
            ]
        except ValueError:
            QMessageBox.critical(self, "Error", "Ingresa valores válidos (1=Si, 2=No, M=1, F=2)")
            return

        result = self.model.predict([patient_data])[0]

        if result == 1:
            self.result_label.setText("Resultado de la predicción: Cáncer de Pulmón")
        else:
            self.result_label.setText("Resultado de la predicción: No Cáncer de Pulmón")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    predictor = LungCancerPredictor()
    predictor.show()
    sys.exit(app.exec_())
