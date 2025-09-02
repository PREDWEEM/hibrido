# Pesos y Sesgos del Modelo ANN – Avefa

Este paquete contiene los archivos de **pesos** y **sesgos** del modelo de red neuronal artificial (ANN) con **65 neuronas en la capa oculta**, listos para ser utilizados desde un repositorio de GitHub.

## Archivos incluidos

- `IW.npy` → Matriz de pesos desde la capa de entrada a la capa oculta (shape: 4×65)
- `bias_IW.npy` → Vector de sesgos de la capa oculta (shape: 65,)
- `LW.npy` → Pesos desde la capa oculta a la capa de salida (shape: 65,)
- `bias_out.npy` → Sesgo de la capa de salida (escalar)

## Uso en la app `app_emergencia.py`

En la barra lateral de la app:
1. Ingresar la **URL base RAW** de tu repositorio de GitHub donde están los `.npy`  
   Ejemplo:
   ```
   https://raw.githubusercontent.com/USUARIO/REPO/RAMA/carpeta_pesos
   ```
2. La app descargará automáticamente `IW.npy`, `bias_IW.npy`, `LW.npy` y `bias_out.npy`.
3. Los datos meteorológicos se cargan aparte (CSV/Excel con columnas: `Julian_days`, `TMIN`, `TMAX`, `Prec`).

## Ejemplo de código para cargar pesos desde GitHub

```python
import numpy as np

base_url = "https://raw.githubusercontent.com/USUARIO/REPO/RAMA/carpeta_pesos"
IW = np.load(f"{base_url}/IW.npy")
bias_IW = np.load(f"{base_url}/bias_IW.npy")
LW = np.load(f"{base_url}/LW.npy")
bias_out = np.load(f"{base_url}/bias_out.npy")
```

## Notas importantes
- Orden de variables de entrada esperado por los pesos: **Julian_days, TMIN, TMAX, Prec**
- Rango de normalización por variable:
  - Julian_days → min=1, max=300
  - TMIN → min=-7, max=25.5
  - TMAX → min=0, max=41
  - Prec → min=0, max=84

## Verificación de integridad
SHA1 primeros 12 caracteres de cada archivo (para comprobar después de subir a GitHub):

| Archivo       | Shape  | SHA1 (12)     |
|---------------|--------|---------------|
| IW.npy        | 4×65   | fc04f0b2a977     |
| bias_IW.npy   | 65     | 6a8d820bb36e |
| LW.npy        | 65     | 091e5acff4e2     |
| bias_out.npy  | escalar| 371dc1d033ec   |

---
**Autor:** Adaptado para proyecto Avefa  
**Fecha:** 2025
