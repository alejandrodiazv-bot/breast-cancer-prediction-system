# Breast Cancer Prediction API

Sistema completo de Machine Learning para predicción de cáncer de mama con API REST, Docker y CI/CD.

## Características

- 🎯 Modelo Random Forest (95.61% accuracy)
- 🌐 API REST con Flask
- 🐳 Contenedor Docker
- 🔄 CI/CD con GitHub Actions
- 📊 Logging y validaciones

## Instalación

### Con Docker
```bash
docker build -t breast-cancer-api .
docker run -p 5000:5000 breast-cancer-api