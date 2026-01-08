"""
----------------------------------------------------------------------------------
PROYECTO: Filtrado de Imágenes con Transformada de Fourier 2D
ASIGNATURA: Matemáticas Avanzadas
AUTORES: Reyes Santamaria Tania Jannet / Castillo Rivera Diego


DESCRIPCIÓN:
Este script implementa un sistema de visión artificial que permite:
1. Cargar imágenes dinámicamente.
2. Aplicar la Transformada Rápida de Fourier (FFT).
3. Filtrar frecuencias mediante máscaras ideales (Pasa-Bajas y Pasa-Altas).
4. Reconstruir la imagen y calcular el error MSE en tiempo real.
----------------------------------------------------------------------------------
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import filedialog
import os
import sys

# ==========================================
# 1. CONFIGURACIÓN DE ENTORNO Y COMPATIBILIDAD
# ==========================================

# Configuración para evitar errores de GUI en macOS (Ventana fantasma)
try:
    root_tk = tk.Tk()
    # Truco: Movemos la ventana de Tkinter fuera de las coordenadas de la pantalla
    # para que sea invisible al usuario pero funcional para el sistema.
    root_tk.geometry("1x1+20000+20000") 
    root_tk.withdraw() 
    root_tk.attributes('-topmost', True) 
except Exception as e:
    print(f"Advertencia del sistema gráfico: {e}")

# Estilo visual de las gráficas (Modo Oscuro para mejor contraste espectral)
plt.style.use('dark_background')
COLOR_SLIDER = '#00d4aa'  # Turquesa neón
COLOR_TEXTO = 'white'

# ==========================================
# 2. FUNCIONES DE PROCESAMIENTO MATEMÁTICO
# ==========================================

def seleccionar_imagen_segura():
    """
    Abre el explorador de archivos nativo del sistema operativo de forma segura.
    Retorna: Ruta del archivo seleccionado o None si se cancela.
    """
    try:
        ruta = filedialog.askopenfilename(
            parent=root_tk,
            title="Selecciona una imagen para analizar",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif")]
        )
        return ruta
    except Exception as e:
        print(f"Error I/O: {e}")
        return None

def cargar_y_preprocesar(ruta):
    """
    Carga la imagen y normaliza sus valores.
    1. Lectura binaria (para soportar caracteres especiales en rutas).
    2. Conversión a Escala de Grises (necesario para Fourier 2D).
    3. Normalización [0, 1].
    """
    img = cv2.imdecode(np.fromfile(ruta, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError("No se pudo decodificar la imagen.")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray / 255.0

def aplicar_fft(imagen):
    """
    Aplica la Transformada Discreta de Fourier 2D.
    Retorna:
        - fshift: Espectro complejo centrado.
        - mag: Magnitud logarítmica para visualización.
    """
    f = np.fft.fft2(imagen)
    fshift = np.fft.fftshift(f) # Mueve la frecuencia cero al centro
    
    # Aplicamos logaritmo para reducir el rango dinámico y visualizar detalles
    mag = 20 * np.log(np.abs(fshift) + 1e-5)
    return fshift, mag

def crear_mascara(shape, radio, tipo):
    """
    Genera una máscara binaria circular (Filtro Ideal).
    Parámetros:
        - shape: Dimensiones (filas, columnas).
        - radio: Frecuencia de corte (D0).
        - tipo: 'pasa_bajas' o 'pasa_altas'.
    """
    filas, cols = shape
    centro_fila, centro_col = filas // 2, cols // 2
    
    # Crear grid de coordenadas (Malla euclidiana)
    y, x = np.ogrid[:filas, :cols]
    distancia = np.sqrt((x - centro_col)**2 + (y - centro_fila)**2)
    
    if tipo == 'pasa_bajas':
        return (distancia <= radio).astype(float)
    else:
        return (distancia > radio).astype(float)

def reconstruir_imagen(fshift, mascara):
    """
    Aplica el Teorema de Convolución (Multiplicación en Frecuencia)
    y retorna la imagen al dominio espacial (IFFT).
    """
    fshift_filtrado = fshift * mascara
    f_ishift = np.fft.ifftshift(fshift_filtrado) # Deshacer el centrado
    img_back = np.fft.ifft2(f_ishift)            # Transformada Inversa
    return np.abs(img_back)                      # Magnitud real

def calcular_mse(orig, recons):
    """Calcula el Error Cuadrático Medio (Mean Squared Error)."""
    return np.mean((orig - recons) ** 2)

# ==========================================
# 3. BLOQUE PRINCIPAL (MAIN LOOP)
# ==========================================
if __name__ == "__main__":
    try:
        print("Esperando selección de imagen...")
        ruta_inicial = seleccionar_imagen_segura()
        
        if not ruta_inicial:
            print("Operación cancelada por el usuario.")
            sys.exit()
            
        # --- Inicialización del Estado Global ---
        img_orig = cargar_y_preprocesar(ruta_inicial)
        fshift, espectro_mag = aplicar_fft(img_orig)
        
        # Diccionario para mantener el estado entre eventos de la GUI
        state = {
            'img': img_orig,
            'fshift': fshift,
            'nombre': os.path.basename(ruta_inicial),
            'plots': {} # Referencias a los objetos gráficos para actualizarlos
        }
        radio_init = 30 # Valor empírico inicial

        # --- Configuración de la Interfaz Gráfica (Matplotlib) ---
        fig = plt.figure(figsize=(16, 9), facecolor='#121212')
        # Márgenes ajustados para dar espacio a los controles inferiores
        plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.20, wspace=0.2, hspace=0.3)
        gs = GridSpec(2, 4, figure=fig)
        
        titulo_fig = fig.suptitle(f'ANÁLISIS DE FOURIER: {state["nombre"]}', 
                                  fontsize=18, color=COLOR_TEXTO, fontweight='bold')

        # Definición de Ejes
        ax_orig = fig.add_subplot(gs[:, 0])
        ax_spec = fig.add_subplot(gs[:, 1])
        ax_m_hp = fig.add_subplot(gs[0, 2])
        ax_r_hp = fig.add_subplot(gs[0, 3])
        ax_m_lp = fig.add_subplot(gs[1, 2])
        ax_r_lp = fig.add_subplot(gs[1, 3])

        def dibujar_todo(radio):
            """
            Función maestra de renderizado.
            Limpia y redibuja todos los ejes. Es crucial para manejar 
            correctamente el cambio de resolución (Zoom/Aspect Ratio).
            """
            img = state['img']
            fs = state['fshift']
            filas, cols = img.shape

            # 1. Limpieza total de ejes
            for ax in [ax_orig, ax_spec, ax_m_hp, ax_r_hp, ax_m_lp, ax_r_lp]:
                ax.clear()
                ax.axis('off')
                
            # 2. Cálculo de Filtros actuales
            m_hp = crear_mascara(img.shape, radio, 'pasa_altas')
            r_hp = reconstruir_imagen(fs, m_hp)
            m_lp = crear_mascara(img.shape, radio, 'pasa_bajas')
            r_lp = reconstruir_imagen(fs, m_lp)

            # 3. Renderizado - Lado Izquierdo (Estático)
            ax_orig.imshow(img, cmap='gray')
            ax_orig.set_title("IMAGEN ORIGINAL", color=COLOR_SLIDER, fontsize=10)
            
            _, mag = aplicar_fft(img) 
            ax_spec.imshow(mag, cmap='magma')
            ax_spec.set_title("ESPECTRO DE FRECUENCIA", color=COLOR_SLIDER, fontsize=10)

            # 4. Renderizado - Lado Derecho (Dinámico)
            # --- Pasa Altas ---
            state['plots']['m_hp'] = ax_m_hp.imshow(m_hp, cmap='gray', vmin=0, vmax=1)
            ax_m_hp.set_title("MÁSCARA PASA-ALTAS", color=COLOR_SLIDER, fontsize=10)
            
            state['plots']['r_hp'] = ax_r_hp.imshow(r_hp, cmap='gray')
            state['plots']['t_hp'] = ax_r_hp.set_title(f"MSE: {calcular_mse(img, r_hp):.5f}", color='white')

            # --- Pasa Bajas ---
            state['plots']['m_lp'] = ax_m_lp.imshow(m_lp, cmap='gray', vmin=0, vmax=1)
            ax_m_lp.set_title("MÁSCARA PASA-BAJAS", color=COLOR_SLIDER, fontsize=10)
            
            state['plots']['r_lp'] = ax_r_lp.imshow(r_lp, cmap='gray')
            state['plots']['t_lp'] = ax_r_lp.set_title(f"MSE: {calcular_mse(img, r_lp):.5f}", color='white')

        # Primer dibujado
        dibujar_todo(radio_init)

        # --- Controles de Usuario (Widgets) ---
        
        # Botón Cargar (Izquierda)
        ax_load = plt.axes([0.05, 0.05, 0.12, 0.04]) 
        btn_load = Button(ax_load, 'Cargar Imagen', color='#0055aa', hovercolor='#0077cc')
        btn_load.label.set_color('white')

        # Slider (Centro)
        filas, cols = state['img'].shape
        ax_slider = plt.axes([0.35, 0.05, 0.40, 0.03], facecolor='#333333') 
        slider = Slider(ax_slider, 'Radio de Corte ', 1, min(filas, cols)//2 - 1, 
                        valinit=radio_init, valstep=1, color=COLOR_SLIDER)
        slider.label.set_color('white')
        slider.valtext.set_color('cyan')

        # Botón Reset (Derecha)
        ax_reset = plt.axes([0.82, 0.05, 0.08, 0.04])
        btn_reset = Button(ax_reset, 'Reset', color='#444444', hovercolor='#666666')
        btn_reset.label.set_color('white')

        # --- Callbacks (Manejadores de Eventos) ---

        def update_slider(val):
            """Actualización ligera: Solo modifica los datos de las imágenes, no los ejes."""
            r = slider.val
            img = state['img']
            fs = state['fshift']
            
            # Recalcular lógica
            nm_hp = crear_mascara(img.shape, r, 'pasa_altas')
            nr_hp = reconstruir_imagen(fs, nm_hp)
            nm_lp = crear_mascara(img.shape, r, 'pasa_bajas')
            nr_lp = reconstruir_imagen(fs, nm_lp)

            # Actualizar visuales
            state['plots']['m_hp'].set_data(nm_hp)
            state['plots']['r_hp'].set_data(nr_hp)
            state['plots']['t_hp'].set_text(f"MSE: {calcular_mse(img, nr_hp):.5f}")

            state['plots']['m_lp'].set_data(nm_lp)
            state['plots']['r_lp'].set_data(nr_lp)
            state['plots']['t_lp'].set_text(f"MSE: {calcular_mse(img, nr_lp):.5f}")
            
            fig.canvas.draw_idle()

        def cargar_nueva(event):
            """Gestiona la carga de una nueva imagen y el reajuste de escala."""
            print("Abriendo selector...")
            nueva_ruta = seleccionar_imagen_segura()
            
            if nueva_ruta:
                try:
                    print(f"Procesando: {os.path.basename(nueva_ruta)}")
                    nueva_img = cargar_y_preprocesar(nueva_ruta)
                    n_fshift, _ = aplicar_fft(nueva_img)
                    
                    # Actualizar estado global
                    state['img'] = nueva_img
                    state['fshift'] = n_fshift
                    state['nombre'] = os.path.basename(nueva_ruta)
                    titulo_fig.set_text(f'ANÁLISIS DE FOURIER: {state["nombre"]}')
                    
                    # Actualizar límites del Slider según nueva resolución
                    f, c = nueva_img.shape
                    slider.valmax = min(f, c) // 2 - 1
                    slider.val = 30 
                    slider.ax.set_xlim(slider.valmin, slider.valmax)
                    
                    # Redibujado completo (Corrige problemas de crop/zoom)
                    dibujar_todo(30)
                    fig.canvas.draw()
                    
                    print("Imagen actualizada correctamente.")
                except Exception as e:
                    print(f"Error al cargar imagen: {e}")

        # Conexión de eventos
        slider.on_changed(update_slider)
        btn_reset.on_clicked(lambda x: slider.reset())
        btn_load.on_clicked(cargar_nueva)

        print("Sistema iniciado. Interfaz gráfica activa.")
        plt.show()

    except Exception as e:
        print(f"Error fatal en ejecución: {e}")