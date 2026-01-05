import numpy as np
import pyvista as pv

#DATOS

r = np.load(r"resultados/posiciones.npz")['pos']

nick_name=["Sol",
           "Mercurio",
           "Venus",
           "Tierra",
           "Marte",
           "Jupiter",
           "Saturno",
           "Urano",
           "Neptuno",
           "Pluton"
           ]

colores=[
    "#FFD700",
    "#1E90FF",
    "#FF4500",
    "#32CD32",
    "#FF8C00",
    "#8A2BE2",
    "#00CED1",
    "#FF00FF",
    "#7FFF00",
    "#FFFFFF"
]

#PLOT
plotter=pv.Plotter()
plotter.background_color="black"
step=50
posición_list_new=np.transpose(r, (1, 2, 0))
for i in range(r.shape[1]):
    nube=pv.PolyData(posición_list_new[i,:,::step].T)
    plotter.add_points(nube, render_points_as_spheres=True, point_size=5, color=colores[i],label=nick_name[i])

    ultimo_punto=pv.PolyData(posición_list_new[i, :, -1].reshape(1, 3))
    plotter.add_points(
        ultimo_punto,
        render_points_as_spheres=True,
        point_size=15,   
        color=colores[i]
    )

plotter.add_text("Sistema Solar - Modelo 10 cuerpos",  position="upper_edge", font_size=14, color="white")  
plotter.add_legend(bcolor=None, size=(0.3, 0.3),loc="lower right")
plotter.show()
