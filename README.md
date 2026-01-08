# PROYECTO: Filtros en Imágenes con Transformada de Fourier 2D (FFT)

**AUTORES:**
* Reyes Santamaria Tania Jannet
* Castillo Rivera Diego

---

##  Descripción
Herramienta interactiva desarrollada en Python para el procesamiento digital de imágenes en el dominio de la frecuencia.

A diferencia de los filtros espaciales convencionales, este proyecto aplica la **Transformada Discreta de Fourier (DFT)** para descomponer la imagen en sus componentes frecuenciales. Permite al usuario aplicar filtros **Pasa-Bajas** (suavizado) y **Pasa-Altas** (detección de bordes) mediante máscaras ideales, visualizando los resultados en tiempo real.

###  Características Principales
* **Interfaz Gráfica "Dark Mode":** Diseño moderno y amigable para la visualización de espectros.
* **Interactividad en Tiempo Real:** Uso de un *Slider* para ajustar el radio de corte (frecuencia) dinámicamente.
* **Carga Dinámica:** Botón dedicado para cambiar de imagen sin reiniciar el programa (con ajuste automático de escala).
* **Métricas:** Cálculo automático del Error Cuadrático Medio (MSE) para evaluar la calidad de reconstrucción.
* **Compatibilidad:** Optimizado para funcionar sin errores en Windows, macOS y Linux.

---

##  Requisitos del Sistema

* **Lenguaje:** Python 3.8 o superior.
* **Sistema Operativo:** Windows, macOS o Linux.

### Librerías Necesarias
El proyecto requiere las siguientes dependencias para cálculo numérico y graficación:

* `opencv-python` (Procesamiento de imagen)
* `numpy` (Cálculo matricial y FFT)
* `matplotlib` (Interfaz gráfica y visualización)
* `tkinter` (Selector de archivos nativo - suele venir con Python)

---

##  Instalación

1. Abra su terminal o Símbolo del Sistema.
2. Ejecute el siguiente comando para instalar las dependencias:

```bash
pip install opencv-python numpy matplotlib
