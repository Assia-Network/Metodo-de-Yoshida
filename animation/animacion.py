import pyvista as pv
from pyvistaqt import BackgroundPlotter
import numpy as np

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

#Cargar datos  (UBICACION)
posciones_recuperadas=np.load(r"resultados/posiciones.npz")["pos"].transpose(1,2,0)[:,:,::50]
print(posciones_recuperadas.shape)

#Numero de Planetas
N_bodies=posciones_recuperadas.shape[0]

#Camtodad de posiciones
n_frames=posciones_recuperadas.shape[2]

#Variables globales
frame_index=0
running=False

#Iniciador de PLotter
plotter=BackgroundPlotter()
plotter.background_color="black"

#Puntos iniciales
nubes=[]
nubesv2=[]
n_rastro=1000
n_frames-=n_rastro
for i in range(N_bodies):
    nube=pv.PolyData(posciones_recuperadas[i,:,n_rastro].T)
    actor=plotter.add_points(nube, render_points_as_spheres=True, point_size=15, color=colores[i],label=nick_name[i])
    nubes.append(nube)
    
    nubev2=pv.PolyData(posciones_recuperadas[i,:,:n_rastro].T)
    actorv2=plotter.add_points(nubev2, render_points_as_spheres=True, point_size=5, color=colores[i], opacity=0.4)
    nubesv2.append(nubev2)
    

#Titulo y Legenda
plotter.add_text("Sistema Solar [0]",  position="upper_edge", font_size=14, color="white", name="titulo")  
plotter.add_legend(bcolor=None, size=(0.3, 0.3),loc="lower right")

#Funcion de actulizacion de frame
def update_frame():
    global frame_index, running
    print("Play/Pause:", running)
    if running:
        frame_index=(frame_index + 1)%n_frames
        for i, nube in enumerate(nubes):
            nube.points=posciones_recuperadas[i, :, frame_index+n_rastro].T
        for i, nubev2 in enumerate(nubesv2):
            nubev2.points=posciones_recuperadas[i, :, frame_index:frame_index+n_rastro].T
        
            
        plotter.remove_actor("titulo")
        plotter.add_text(f"Sistema Solar [{frame_index}]",  position="upper_edge", font_size=14, color="white", name="titulo")
        plotter.render()

#Play o Pause
def toggle_play():
    global running
    running=not running
    update_frame()
    
#Reset de animacion
def reset_animation():
    global frame_index
    frame_index=0
    for i, nube in enumerate(nubes):
        nube.points=posciones_recuperadas[i, :, frame_index].T
        
    plotter.remove_actor("titulo")
    plotter.add_text(f"Sistema Solar [{frame_index}]",  position="upper_edge", font_size=14, color="white", name="titulo") 
    plotter.render()
    
#Comandos
plotter.add_key_event("s", toggle_play)  #stop
plotter.add_key_event("r", reset_animation)  #reset

#Loop
plotter.add_callback(update_frame, interval=1)

#mostrar
plotter.app.exec_()

#inportante
# pip install numpy pyvista pyvistaqt qtpy
# pip install ipywidgets jupyter_bokeh panel
