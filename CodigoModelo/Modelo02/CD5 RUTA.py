# vrp_cd5_print_fixed.py 
# ORS Matrix + MILP (PuLP) para CD5 — Corrección de nombres de variables
import os
import math
import time
import requests
import pulp
import folium
import polyline
import numpy as np
import pandas as pd

# ---------------------------
# CONFIG
# ---------------------------
# NOTA: Ten cuidado al compartir este código públicamente con tu API Key real.
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjhhOWE5NmYxMzZiNjQ5OWI5MTgxMDEyMTUzMDJhYjk4IiwiaCI6Im11cm11cjY0In0="
PROFILE = "driving-car"
MATRIX_URL = f"https://api.openrouteservice.org/v2/matrix/{PROFILE}"
DIRECTIONS_URL = f"https://api.openrouteservice.org/v2/directions/{PROFILE}/geojson"

# parámetros operativos
Q = 2000                    # capacidad (kg)
SERVICE_SEC = 30 * 60       # 30 min en segundos (descarga)
RECHARGE_SEC = 30 * 60      # 30 min en segundos (recarga en CD por partida)

# ---------------------------
# DATOS (CD5 + comedores)
# ---------------------------
# formato: (id, kg, lat, lon)
rows = [
    ("C24",  340, -12.209152, -76.954383),
    ("C33",  340, -12.207971, -76.955091),
    ("C39",  327, -12.209522, -76.955152),
    ("C64",  465, -12.201579, -76.954474),
    ("C89",  465, -12.203056, -76.954142),
    ("C114", 327, -12.198312, -76.956683),
    ("C121", 327, -12.208202, -76.953203),
    ("C130", 327, -12.206564, -76.951343),
    ("C140", 327, -12.203154, -76.952724),
]

# coordenadas exactas del CD5
CD5_LAT = -12.208121
CD5_LON = -76.953076

# ---------------------------
# Construir nodos: nodo 0 = CD, luego comedores
# ---------------------------
nodes = []
nodes.append({"id":"CD5","q":0,"lat":CD5_LAT,"lon":CD5_LON})
for r in rows:
    nodes.append({"id": r[0], "q": float(r[1]), "lat": r[2], "lon": r[3]})

N = len(nodes)
print("N nodos (incluyendo CD):", N)

# preparar locations para ORS: [lon, lat]
locations = [[n["lon"], n["lat"]] for n in nodes]

# ---------------------------
# Llamar ORS Matrix (distance m, duration s)
# ---------------------------
if ORS_API_KEY.strip() == "":
    raise SystemExit("Pon tu ORS_API_KEY en la variable ORS_API_KEY.")

payload = {
    "locations": locations,
    "metrics": ["distance", "duration"],
    "units": "m"
}
headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}

print("Solicitando matriz ORS...")
resp = requests.post(MATRIX_URL, json=payload, headers=headers)
if resp.status_code != 200:
    print("Error ORS matrix:", resp.status_code, resp.text)
    raise SystemExit("ORS matrix failed")

mat = resp.json()
dist_m = mat["distances"]    # metros
time_s = mat["durations"]    # segundos

# ---------------------------
# Preparar MILP
# ---------------------------
total_demand = sum(n["q"] for n in nodes)
K_min = math.ceil(total_demand / Q)   # mínimo de viajes necesario
K_max = K_min + 1                     # permitir hasta 1 viaje extra
print("Demanda total (kg):", total_demand, "=> K_min =", K_min, ", K_max =", K_max)

I = list(range(N))
Ks = list(range(K_max))  # Ks = [0, 1, ..., K_max-1]

model = pulp.LpProblem("VRP_multitrip", pulp.LpMinimize)

# --- VARIABLES ---
# x[i,j,k]
x = {}
for i in I:
    for j in I:
        if i == j: continue
        for k in Ks:
            # CORRECCIÓN AQUÍ: Agregamos guiones bajos para que el nombre sea único
            # Antes: f"x_{i}{j}{k}" -> Error de duplicados (ej: 1-12 vs 11-2)
            # Ahora: f"x_{i}_{j}_{k}" -> Correcto
            x[(i,j,k)] = pulp.LpVariable(f"x_{i}_{j}_{k}", cat="Binary")

# MTZ u[i,k] (solo para i>0)
u = {}
for i in I[1:]:
    for k in Ks:
        u[(i,k)] = pulp.LpVariable(f"u_{i}_{k}", lowBound=0, upBound=Q, cat="Continuous")

# --- FUNCION OBJETIVO ---
# minimizar distancia total (m)
model += pulp.lpSum(dist_m[i][j] * x[(i,j,k)] for i in I for j in I if i!=j for k in Ks)

# --- RESTRICCIONES ---

# 1. Cada comedor visitado exactamente 1 vez
for i in I[1:]:
    model += pulp.lpSum(x[(i,j,k)] for j in I if j!=i for k in Ks) == 1

# 2. Flujo por viaje (entrar = salir)
for k in Ks:
    for i in I:
        model += pulp.lpSum(x[(i,j,k)] for j in I if j!=i) == pulp.lpSum(x[(j,i,k)] for j in I if j!=i)

# 3. Cada viaje inicia y termina en depósito (<=1 para permitir viajes vacíos)
for k in Ks:
    model += pulp.lpSum(x[(0,j,k)] for j in I if j!=0) <= 1
    model += pulp.lpSum(x[(i,0,k)] for i in I if i!=0) <= 1

# 4. MTZ capacidad accumulation & subtour elimination
for k in Ks:
    for i in I[1:]:
        for j in I[1:]:
            if i==j: continue
            model += u[(i,k)] - u[(j,k)] + Q * x[(i,j,k)] <= Q - nodes[j]["q"]
    for i in I[1:]:
        model += u[(i,k)] >= nodes[i]["q"]
        model += u[(i,k)] <= Q

# ---------------------------
# Resolver
# ---------------------------
print("Resolviendo MILP (PuLP CBC)... esto puede tardar un poco...")
solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=300)
res = model.solve(solver)
print("Status:", pulp.LpStatus[model.status])

if pulp.LpStatus[model.status] != "Optimal" and pulp.LpStatus[model.status] != "Optimal":
    print("No óptimo o no solucionado completamente. Estado:", pulp.LpStatus[model.status])

# ---------------------------
# Reconstruir rutas por viaje (walk arcs)
# ---------------------------
routes = []   # lista de (k, route_list)
for k in Ks:
    # buscar salida del deposito
    outs = [j for j in I if j!=0 and pulp.value(x[(0,j,k)]) is not None and pulp.value(x[(0,j,k)]) > 0.5]
    if not outs:
        continue
    
    # puede haber solo 1 salida; empezamos por ella
    route = [0]
    cur = outs[0]
    route.append(cur)
    
    # caminamos los arcos
    steps = 0
    max_steps = N + 2 # seguridad
    while steps < max_steps:
        steps += 1
        found = False
        for j in I:
            if j == cur: continue
            val = pulp.value(x.get((cur,j,k), None))
            if val is not None and val > 0.5:
                route.append(j)
                cur = j
                found = True
                break
        
        if not found:
            # si no hay siguiente, forzar cierre si no es 0 (aunque el modelo debería garantizarlo)
            if route[-1] != 0:
                route.append(0)
            break
        
        if cur == 0:
            break
            
    routes.append((k, route))

# ---------------------------
# Calcular estadísticos y formatear salida
# ---------------------------
total_dist_m = 0
total_time_s = 0
total_delivered = 0

print("\n--- RESULTADO DE RUTAS ---") 
for idx, (k, route) in enumerate(routes, start=1):
    cum_load = 0
    cum_time = 0
    route_dist_m = 0
    print(f"Ruta {idx} (Viaje ID {k}):")
    # inicio
    print(f" CD5 (Carga: {cum_load}, Tiempo: {cum_time} seg)", end="")

    for pos in range(len(route)-1):
        i = route[pos]
        j = route[pos+1]
        
        d_ij = dist_m[i][j]
        t_ij = time_s[i][j]
        
        route_dist_m += d_ij
        cum_time += int(t_ij)
        
        # si llegamos a comedor j
        if j != 0:
            cum_load += int(nodes[j]["q"])
            cum_time += SERVICE_SEC
            print(f" -> {nodes[j]['id']} (Carga: {cum_load}, Tiempo: {cum_time} seg)", end="")
        else:
            # regreso al CD
            # Sumamos recarga si no es la última ruta (opcional, aquí lo sumamos siempre al final del viaje para contabilizarlo)
            cum_time += RECHARGE_SEC
            print(f" -> CD5 (Llegada+Recarga. Tiempo acum: {cum_time} seg)", end="")

    print("\nDistancia: {} m".format(int(route_dist_m)))
    print("Carga total: {} kg".format(int(cum_load)))
    print("Tiempo total viaje: {} seg\n".format(int(cum_time)))

    total_dist_m += route_dist_m
    total_time_s += cum_time
    total_delivered += cum_load

# totales
print("---------------------------")
print("Distancia total flota: {} m".format(int(total_dist_m)))
print("Tiempo total flota (con esperas): {} segundos".format(int(total_time_s)))
print("Total entregado: {} kg".format(int(total_delivered)))

# ---------------------------
# Generar mapa con ORS directions
# ---------------------------
m = folium.Map(location=[CD5_LAT, CD5_LON], zoom_start=14)
folium.Marker([CD5_LAT, CD5_LON], popup="CD5", icon=folium.Icon(color="red", icon="truck")).add_to(m)
for n in nodes[1:]:
    folium.Marker([n["lat"], n["lon"]], popup=f"{n['id']} ({int(n['q'])})", icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)

colors = ["green","purple","orange","brown","gray","olive","cyan","magenta","black"]

for r_idx, (k, route) in enumerate(routes):
    coords = []
    for node_idx in route:
        coords.append([nodes[node_idx]["lon"], nodes[node_idx]["lat"]])
    
    # pedir directions
    try:
        rr = requests.post(DIRECTIONS_URL, json={"coordinates": coords}, headers=headers)
        color = colors[r_idx % len(colors)]
        
        if rr.status_code != 200:
            # fallback: líneas rectas
            latlons = [[nodes[i]["lat"], nodes[i]["lon"]] for i in route]
            folium.PolyLine(latlons, color=color, weight=4, opacity=0.6, popup=f"Ruta {r_idx+1} (Directa)").add_to(m)
        else:
            gj = rr.json()
            for feat in gj.get("features", []):
                geom = feat.get("geometry", {})
                if geom.get("type") == "LineString":
                    coords_line = geom.get("coordinates")
                    # geojson es [lon, lat], folium quiere [lat, lon]
                    latlon = [[c[1], c[0]] for c in coords_line]
                    folium.PolyLine(latlon, color=color, weight=4, opacity=0.7, popup=f"Ruta {r_idx+1} (ORS)").add_to(m)
        time.sleep(1) # cortesía API
    except Exception as e:
        print(f"Error dibujando ruta {r_idx+1}: {e}")
        latlons = [[nodes[i]["lat"], nodes[i]["lon"]] for i in route]
        folium.PolyLine(latlons, color="black", weight=2, opacity=0.5).add_to(m)

outmap = "mapa_CD5.html"
m.save(outmap)
print("\nMapa guardado en:", outmap)