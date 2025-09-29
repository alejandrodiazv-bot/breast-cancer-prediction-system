import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer
import joblib
import os
import logging
import sys

# Crear directorios necesarios
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def load_breast_cancer_dataset():
    """Carga el dataset de breast cancer directamente desde scikit-learn"""
    try:
        logger.info("📂 Cargando dataset de Breast Cancer desde scikit-learn...")
        
        # Cargar dataset integrado
        cancer_data = load_breast_cancer()
        
        # Crear DataFrame con las características
        df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
        
        # Añadir la columna diagnosis (0=Benigno, 1=Maligno)
        df['diagnosis'] = cancer_data.target
        # Convertir 0/1 a B/M (B=Benign, M=Malignant)
        df['diagnosis'] = df['diagnosis'].map({0: 'B', 1: 'M'})
        
        logger.info(f"✅ Dataset cargado exitosamente")
        logger.info(f"📊 Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")
        logger.info(f"🔢 Distribución de clases: {df['diagnosis'].value_counts().to_dict()}")
        logger.info(f"🎯 Significado: B=Benigno (No canceroso), M=Maligno (Canceroso)")
        
        return df, cancer_data.feature_names.tolist(), cancer_data.target_names.tolist()
        
    except Exception as e:
        logger.error(f"❌ Error cargando dataset: {e}")
        raise

def preprocess_data(df, feature_names):
    """Preprocesa los datos para el entrenamiento"""
    try:
        logger.info("🔧 Preprocesando datos...")
        
        # Verificar que existe la columna diagnosis
        if 'diagnosis' not in df.columns:
            raise ValueError(f"Columna 'diagnosis' no encontrada. Columnas disponibles: {df.columns.tolist()}")
        
        # Codificar variable objetivo (B=0, M=1)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['diagnosis'])
        
        # Seleccionar características
        X = df[feature_names]
        
        logger.info(f"✅ Preprocesamiento completado")
        logger.info(f"📊 Características (X): {X.shape}")
        logger.info(f"📊 Objetivo (y): {y.shape}")
        logger.info(f"🔢 Mapping diagnóstico: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
        
        return X, y, label_encoder
        
    except Exception as e:
        logger.error(f"❌ Error en preprocesamiento: {e}")
        raise

def train_and_evaluate_model():
    """Entrena y evalúa el modelo de clasificación de cáncer de mama"""
    try:
        print("=" * 70)
        print("🚀 INICIANDO ENTRENAMIENTO - MODELO DE DETECCIÓN DE CÁNCER DE MAMA")
        print("=" * 70)
        
        # 1. Cargar datos desde scikit-learn
        df, feature_names, target_names = load_breast_cancer_dataset()
        
        # 2. Preprocesar datos
        X, y, label_encoder = preprocess_data(df, feature_names)
        
        # 3. Dividir datos (80% entrenamiento, 20% prueba)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y,
            shuffle=True
        )
        
        logger.info(f"📊 División de datos:")
        logger.info(f"   - Entrenamiento: {X_train.shape[0]} muestras")
        logger.info(f"   - Prueba: {X_test.shape[0]} muestras")
        
        # 4. Escalar características (normalización)
        logger.info("⚖️ Escalando características...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 5. Entrenar modelo Random Forest
        logger.info("🤖 Entrenando modelo Random Forest...")
        model = RandomForestClassifier(
            n_estimators=100,      # Número de árboles
            max_depth=10,          # Profundidad máxima
            random_state=42,       # Semilla para reproducibilidad
            n_jobs=-1              # Usar todos los cores disponibles
        )
        
        model.fit(X_train_scaled, y_train)
        logger.info("✅ Modelo entrenado exitosamente")
        
        # 6. Evaluar el modelo
        logger.info("📈 Evaluando modelo...")
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 7. Mostrar resultados detallados
        print("\n" + "=" * 70)
        print("🎯 RESULTADOS DEL MODELO DE CLASIFICACIÓN")
        print("=" * 70)
        print(f"✅ EXACTITUD (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"📊 REPORTE DE CLASIFICACIÓN:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        print(f"📋 MATRIZ DE CONFUSIÓN:")
        print("       Predicción")
        print("       B     M")
        print(f"Real B {cm[0,0]:<5} {cm[0,1]:<5}")
        print(f"     M {cm[1,0]:<5} {cm[1,1]:<5}")
        
        # 8. Guardar artefactos del modelo
        logger.info("💾 Guardando modelo y artefactos...")
        
        # Guardar cada componente por separado
        artifacts = {
            'model': model,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'feature_names': feature_names,
            'target_names': target_names
        }
        
        for name, artifact in artifacts.items():
            joblib.dump(artifact, f'models/{name}.joblib')
            logger.info(f"   ✅ {name}.joblib guardado")
        
        # 9. Guardar metadata del modelo
        metadata = {
            'accuracy': float(accuracy),
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'classes': label_encoder.classes_.tolist(),
            'class_mapping': dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))),
            'model_type': 'RandomForestClassifier',
            'version': '1.0.0',
            'training_date': pd.Timestamp.now().isoformat(),
            'dataset_info': {
                'samples': len(df),
                'features': len(feature_names),
                'class_distribution': df['diagnosis'].value_counts().to_dict()
            }
        }
        
        joblib.dump(metadata, 'models/metadata.joblib')
        
        # 10. Resumen final
        print("\n" + "=" * 70)
        print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("=" * 70)
        print(f"📍 Modelo guardado en: models/")
        print(f"🎯 Exactitud del modelo: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"🔢 Número de características: {len(feature_names)}")
        print(f"📊 Distribución de clases: {df['diagnosis'].value_counts().to_dict()}")
        print(f"💾 Archivos guardados:")
        for name in artifacts.keys():
            print(f"   - {name}.joblib")
        print(f"   - metadata.joblib")
        
        return model, accuracy, metadata
        
    except Exception as e:
        logger.error(f"❌ Error durante el entrenamiento: {e}")
        print(f"❌ ERROR: {e}")
        raise

if __name__ == "__main__":
    train_and_evaluate_model()