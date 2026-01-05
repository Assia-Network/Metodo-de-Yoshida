# Método de Yoshida: Simulación Gravitacional de 10 Cuerpos

Este repositorio contiene una implementación del **Método de Yoshida**, un integrador simpléctico de alto orden diseñado para la simulación precisa de sistemas Hamiltonianos. Se aplica aquí para resolver la dinámica de un sistema gravitacional de 10 cuerpos en interacción mutua.



## Descripción Teórica
A diferencia de los métodos de integración estándar (como Runge-Kutta), el algoritmo de Yoshida preserva la estructura simpléctica del espacio de fases. Esto garantiza que errores en la energía total del sistema no crezcan linealmente con el tiempo, permitiendo integraciones estables en simulaciones de largo plazo.

### Características principales:
* **Integrador Simpléctico:** Implementación de 4º orden.
* **Estabilidad Térmica:** Conservación del Hamiltoniano (Energía total).
* **N-Body Problem:** Dinámica orbital de 10 cuerpos con interacción gravitatoria.
* **Visualización:** Herramientas para generar animaciones 3D de las trayectorias.

---

## Generación de Datos
Los archivos de resultados no se incluyen en el repositorio debido a su tamaño (~1 GB en total).

> **Instrucciones:** Para realizar las visualizaciones y animaciones, ejecuta primero el notebook `main.ipynb`. Este generará automáticamente los archivos `momentos.npz` y `posiciones.npz` en el directorio resultados. Una vez generados, los scripts de visualización los detectarán de forma automática.

---

##  Requisitos
Para ejecutar este proyecto, necesitas tener instalado:
* Python 3.x
* NumPy
* Matplotlib (para las animaciones y gráficas)
* PyVista (para la visualización 3D)

```python
# Ejemplo rápido para cargar los datos una vez generados
import numpy as np
posiciones = np.load('posiciones.npz')
print(posiciones.files)
