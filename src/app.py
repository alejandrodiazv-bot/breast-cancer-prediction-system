from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import logging
import os
import sys
from datetime import datetime

# Crear directorios necesarios
os.makedirs('logs', exist_ok=True)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('BreastCancerAPI')

app = Flask(__name__)

# Cargar modelo y artefactos
try:
    model = joblib.load('models/model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    label_encoder = joblib.load('models/label_encoder.joblib')
    feature_names = joblib.load('models/feature_names.joblib')
    metadata = joblib.load('models/metadata.joblib')
    
    logger.info("✅ Modelo y artefactos cargados exitosamente")
    logger.info(f"📊 Modelo: {metadata['model_type']} v{metadata['version']}")
    logger.info(f"🎯 Accuracy: {metadata['accuracy']:.4f}")
    logger.info(f"🔢 Features: {metadata['n_features']}")
    
except Exception as e:
    logger.error(f"❌ Error cargando el modelo: {e}")
    model = scaler = label_encoder = feature_names = metadata = None

class ValidationError(Exception):
    pass

def validate_prediction_input(data):
    """Valida los datos de entrada para predicción"""
    if not isinstance(data, list):
        raise ValidationError("Los datos deben ser una lista de registros")
    
    if len(data) == 0:
        raise ValidationError("La lista de registros no puede estar vacía")
    
    errors = []
    for i, record in enumerate(data):
        if not isinstance(record, dict):
            errors.append(f"Registro {i}: debe ser un objeto JSON")
            continue
            
        # Verificar características faltantes
        missing_features = set(feature_names) - set(record.keys())
        if missing_features:
            errors.append(f"Registro {i}: características faltantes: {list(missing_features)[:5]}...")
            
        # Verificar tipos de datos
        for feature in feature_names:
            if feature in record:
                if not isinstance(record[feature], (int, float)):
                    errors.append(f"Registro {i}: {feature} debe ser numérico")
    
    if errors:
        raise ValidationError("; ".join(errors[:10]))

@app.route('/', methods=['GET'])
def health_check():
    """Endpoint de verificación de estado del servicio"""
    status = {
        "status": "healthy" if model is not None else "unhealthy",
        "service": "Breast Cancer Prediction API",
        "version": metadata['version'] if metadata else "unknown",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat(),
        "model_info": {
            "type": metadata.get('model_type', 'unknown'),
            "accuracy": metadata.get('accuracy', 0),
            "n_features": metadata.get('n_features', 0),
            "classes": metadata.get('classes', [])
        } if metadata else None
    }
    return jsonify(status)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de predicción de cáncer de mama"""
    start_time = datetime.now()
    
    try:
        # Validar contenido JSON
        if not request.is_json:
            logger.warning("Intento de predicción sin JSON")
            return jsonify({"error": "Content-Type debe ser application/json"}), 400
        
        data = request.get_json()
        
        # Validar estructura básica
        if 'data' not in data:
            return jsonify({"error": "Campo 'data' requerido"}), 400
        
        input_data = data['data']
        
        # Validar datos de entrada
        validate_prediction_input(input_data)
        
        # Verificar que el modelo esté cargado
        if model is None:
            logger.error("Intento de predicción con modelo no cargado")
            return jsonify({"error": "Servicio no disponible. Modelo no cargado."}), 503
        
        # Convertir a DataFrame y asegurar orden de columnas
        df = pd.DataFrame(input_data)
        df = df.reindex(columns=feature_names, fill_value=0)
        
        # Preprocesar y predecir
        scaled_data = scaler.transform(df)
        predictions = model.predict(scaled_data)
        probabilities = model.predict_proba(scaled_data)
        
        # Decodificar predicciones
        diagnosis_labels = label_encoder.inverse_transform(predictions)
        
        # Preparar respuesta
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            confidence = float(max(prob))
            diagnosis = "Malignant" if diagnosis_labels[i] == 'M' else "Benign"
            
            results.append({
                "record_id": i,
                "diagnosis": diagnosis,
                "diagnosis_code": diagnosis_labels[i],
                "confidence": round(confidence, 4),
                "probabilities": {
                    "benign": round(float(prob[0]), 4),
                    "malignant": round(float(prob[1]), 4)
                },
                "risk_level": "high" if diagnosis == "Malignant" and confidence > 0.7 else "medium" if confidence > 0.5 else "low"
            })
        
        # Log de la predicción
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Predicción exitosa - Registros: {len(results)}, Tiempo: {processing_time:.3f}s")
        
        return jsonify({
            "predictions": results,
            "metadata": {
                "model_version": metadata['version'],
                "processing_time_seconds": round(processing_time, 3),
                "records_processed": len(results),
                "timestamp": datetime.utcnow().isoformat()
            }
        })
        
    except ValidationError as e:
        logger.warning(f"Validación fallida: {str(e)}")
        return jsonify({"error": f"Datos de entrada inválidos: {str(e)}"}), 400
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        return jsonify({"error": "Error interno del servidor"}), 500

@app.route('/features', methods=['GET'])
def get_features():
    """Endpoint para obtener la lista de características requeridas"""
    if feature_names is None:
        return jsonify({"error": "Modelo no cargado"}), 503
        
    return jsonify({
        "feature_names": feature_names,
        "count": len(feature_names),
        "description": "Características requeridas para la predicción"
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint no encontrado"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Error interno del servidor"}), 500

if __name__ == '__main__':
    logger.info("🚀 Iniciando Breast Cancer Prediction API")
    print("=" * 60)
    print("🏥 BREAST CANCER PREDICTION API")
    print("=" * 60)
    print("📊 Modelo cargado:", "✅" if model is not None else "❌")
    if model is not None:
        print(f"🎯 Accuracy: {metadata['accuracy']:.2%}")
        print(f"🔢 Características: {len(feature_names)}")
        print(f"🎯 Clases: {metadata['classes']}")
    print("🌐 Servidor iniciado en: http://localhost:5000")
    print("⏹️  Presiona Ctrl+C para detener el servidor")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False)