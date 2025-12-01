# vrp_cd2_print.py 
# ORS Matrix + MILP (PuLP) para CD2 — imprime rutas con carga y tiempo acumulado (seg)
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
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjhhOWE5NmYxMzZiNjQ5OWI5MTgxMDEyMTUzMDJhYjk4IiwiaCI6Im11cm11cjY0In0="  # <- PON TU KEY
PROFILE = "driving-car"
MATRIX_URL = f"https://api.openrouteservice.org/v2/matrix/{PROFILE}"
DIRECTIONS_URL = f"https://api.openrouteservice.org/v2/directions/{PROFILE}/geojson"

# parámetros operativos
Q = 2000                    # capacidad (kg) <-- ACTUALIZADO a la capacidad del CD2 proporcionada
SERVICE_SEC = 30 * 60       # 30 min en segundos (descarga)
RECHARGE_SEC = 30 * 60      # 30 min en segundos (recarga en CD por partida)

# ---------------------------
# DATOS (CD2 + comedores) — pega tus datos aquí
# ---------------------------
# formato: (id, kg, lat, lon)
rows = [
    ("C3",   402, -12.20732,  -76.941006),
    ("C6",   263, -12.216099, -76.928893),
    ("C9",   328, -12.213511, -76.941365),
    ("C12",  464, -12.210318, -76.934330),
    ("C19",  263, -12.217377, -76.930534),
    ("C22",  465, -12.215616, -76.934469),
    ("C23",  326, -12.217434, -76.928081),
    ("C25",  402, -12.206931, -76.935798),
    ("C26",  265, -12.210953, -76.941605),
    ("C31",  327, -12.216103, -76.932664),
    ("C32",  465, -12.204266, -76.936825),
    ("C40",  327, -12.204909, -76.939234),
    ("C44",  340, -12.202893, -76.938586),
    ("C47",  465, -12.208532, -76.937521),
    ("C50",  201, -12.203925, -76.940315),
    ("C56",  265, -12.207671, -76.934920),
    ("C63",  402, -12.203829, -76.941482),
]

# coordenadas exactas del CD2 (provistas)
CD2_LAT = -12.207688
CD2_LON = -76.936671

# ---------------------------
# Construir nodos: nodo 0 = CD, luego comedores
# ---------------------------
nodes = []
nodes.append({"id":"CD2","q":0,"lat":CD2_LAT,"lon":CD2_LON})
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
K_max = K_min + 3                     # permitir hasta 3 viajes extra
print("Demanda total (kg):", total_demand, "=> K_min =", K_min, ", K_max =", K_max)

I = list(range(N))
Ks = list(range(K_max))  # Ks = [0, 1, ..., K_max-1]

model = pulp.LpProblem("VRP_multitrip", pulp.LpMinimize)

# variables x[i,j,k]
x = {}
for i in I:
    for j in I:
        if i==j: continue
        for k in Ks:
            x[(i,j,k)] = pulp.LpVariable(f"x_{i}_{j}_{k}", cat="Binary")

# MTZ u[i,k] (solo para i>0)
u = {}
for i in I[1:]:
    for k in Ks:
        u[(i,k)] = pulp.LpVariable(f"u_{i}_{k}", lowBound=0, upBound=Q, cat="Continuous")

# objetivo: minimizar distancia total (m)
model += pulp.lpSum(dist_m[i][j] * x[(i,j,k)] for i in I for j in I if i!=j for k in Ks)

# restricción: cada comedor visitado exactamente 1 vez
for i in I[1:]:
    model += pulp.lpSum(x[(i,j,k)] for j in I if j!=i for k in Ks) == 1

# flujo por viaje
for k in Ks:
    for i in I:
        model += pulp.lpSum(x[(i,j,k)] for j in I if j!=i) == pulp.lpSum(x[(j,i,k)] for j in I if j!=i)

# cada viaje inicia y termina en depósito (<=1 para permitir viajes vacíos)
for k in Ks:
    model += pulp.lpSum(x[(0,j,k)] for j in I if j!=0) <= 1
    model += pulp.lpSum(x[(i,0,k)] for i in I if i!=0) <= 1

# MTZ capacidad accumulation & subtour elimination
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
routes = []   # lista de (k, route_list_of_nodes_in_order_start0_end0)
for k in Ks:
    # si viaje no usado (no outgoing from depot), skip
    outs = [j for j in I if j!=0 and pulp.value(x[(0,j,k)]) is not None and pulp.value(x[(0,j,k)])>0.5]
    if not outs:
        continue
    # puede haber solo 1 salida; empezamos por ella
    route = [0]
    cur = outs[0]
    route.append(cur)
    while True:
        # buscar siguiente j con x[cur,j,k]==1
        found = False
        for j in I:
            if j==cur: continue
            val = pulp.value(x.get((cur,j,k), None))
            if val is not None and val > 0.5:
                route.append(j)
                cur = j
                found = True
                break
        if not found:
            # si no hay siguiente, terminar intentando volver a depot si no está
            if route[-1] != 0:
                route.append(0)
            break
        if cur == 0:
            break
    routes.append((k, route))

# ---------------------------
# Calcular estadísticos y formatear salida como ejemplo
# ---------------------------
total_dist_m = 0
total_time_s = 0
total_delivered = 0

print("\n") 
for idx, (k, route) in enumerate(routes, start=1):
    cum_load = 0
    cum_time = 0
    route_dist_m = 0
    print(f"Ruta  {idx}:")
    # print starting CD
    print(f" CD2 (Carga: {cum_load}, Tiempo: {cum_time} seg)", end="")

    for pos in range(len(route)-1):
        i = route[pos]
        j = route[pos+1]
        # sumar distancia y tiempo de viaje
        d_ij = dist_m[i][j]
        t_ij = time_s[i][j]
        route_dist_m += d_ij
        cum_time += int(t_ij)            # tiempo de viaje en segundos
        # si llegamos a comedor j
        if j != 0:
            cum_load += int(nodes[j]["q"])
            # al llegar, añadir tiempo de descarga (30 min)
            cum_time += SERVICE_SEC
            print(f" -> {nodes[j]['id']} (Carga: {cum_load}, Tiempo: {cum_time} seg)", end="")
        else:
            # regreso al CD: opcional no sumar recarga aquí (recarga se aplica al iniciar siguiente viaje)
            print(f" -> CD2 (Tiempo: {cum_time} seg)", end="")

    # fin ruta
    print("\nDistancia: {} m".format(int(route_dist_m)))
    print("Carga total: {} kg".format(int(cum_load)))
    print("Tiempo total: {} seg\n".format(int(cum_time)))

    total_dist_m += route_dist_m
    total_time_s += cum_time
    total_delivered += cum_load

# totales
print("Distancia total: {} m".format(int(total_dist_m)))
print("Tiempo total estimado: {} segundos".format(int(total_time_s)))
print("Total entregado: {} kg".format(int(total_delivered)))

# ---------------------------
# Generar mapa con ORS directions para cada ruta
# ---------------------------
m = folium.Map(location=[CD2_LAT, CD2_LON], zoom_start=14)
# marcadores
folium.Marker([CD2_LAT, CD2_LON], popup="CD2", icon=folium.Icon(color="red", icon="truck")).add_to(m)
for n in nodes[1:]:
    folium.Marker([n["lat"], n["lon"]], popup=f"{n['id']} (q={int(n['q'])})", icon=folium.Icon(color="blue")).add_to(m)

colors = ["green","purple","orange","brown","gray","olive","cyan","magenta","black"]
for r_idx, (k, route) in enumerate(routes):
    coords = []
    for node_idx in route:
        coords.append([nodes[node_idx]["lon"], nodes[node_idx]["lat"]])
    # pedir directions
    try:
        rr = requests.post(DIRECTIONS_URL, json={"coordinates": coords}, headers=headers)
        if rr.status_code != 200:
            # fallback: dibujar líneas directas
            latlons = [[nodes[i]["lat"], nodes[i]["lon"]] for i in route]
            folium.PolyLine(latlons, color=colors[r_idx%len(colors)], weight=4, opacity=0.6, popup=f"Viaje {r_idx+1}").add_to(m)
        else:
            gj = rr.json()
            for feat in gj.get("features", []):
                geom = feat.get("geometry", {})
                if geom.get("type") == "LineString":
                    coords_line = geom.get("coordinates")
                    latlon = [[c[1], c[0]] for c in coords_line]
                    folium.PolyLine(latlon, color=colors[r_idx%len(colors)], weight=4, opacity=0.7, popup=f"Viaje {r_idx+1}").add_to(m)
        time.sleep(1)
    except Exception as e:
        latlons = [[nodes[i]["lat"], nodes[i]["lon"]] for i in route]
        folium.PolyLine(latlons, color=colors[r_idx%len(colors)], weight=4, opacity=0.6, popup=f"Viaje {r_idx+1}").add_to(m)

outmap = "mapa_CD2.html"
m.save(outmap)
print("\nMapa guardado en:", outmap)
