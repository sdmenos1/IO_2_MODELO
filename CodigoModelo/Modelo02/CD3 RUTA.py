# vrp_cd3_ortools_map.py
import requests
import folium
import math
import time
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# ---------------------------
# CONFIG
# ---------------------------
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjhhOWE5NmYxMzZiNjQ5OWI5MTgxMDEyMTUzMDJhYjk4IiwiaCI6Im11cm11cjY0In0="  # Pon tu API Key de ORS
PROFILE = "driving-car"

# parámetros operativos
Q = 2000                    # capacidad (kg)
SERVICE_SEC = 30 * 60       # 30 min en segundos (descarga)
RECHARGE_SEC = 30 * 60      # 30 min en segundos (recarga en CD por viaje)
SPEED_KMH = 30              # velocidad camión km/h

# ---------------------------
# DATOS (CD3 + comedores)
# ---------------------------
rows = [
    ("C4", 327, -12.220941, -76.934228),
    ("C5", 465, -12.22038, -76.930412),
    ("C7", 402, -12.224928, -76.929422),
    ("C8", 340, -12.221842, -76.942382),
    ("C10", 201, -12.223859, -76.93273),
    ("C11", 326, -12.224404, -76.928796),
    ("C14", 263, -12.227425, -76.935243),
    ("C15", 201, -12.224324, -76.937127),
    ("C16", 464, -12.222502, -76.931024),
    ("C17", 465, -12.221013, -76.937749),
    ("C18", 201, -12.227254, -76.933929),
    ("C20", 265, -12.229279, -76.931166),
    ("C21", 201, -12.218914, -76.940158),
    ("C27", 465, -12.224027, -76.935597),
    ("C34", 327, -12.22286, -76.933621),
    ("C37", 465, -12.227225, -76.935647),
    ("C41", 340, -12.225716, -76.936778),
    ("C42", 327, -12.219265, -76.938152),
    ("C46", 265, -12.218641, -76.93802),
    ("C48", 327, -12.224881, -76.935366),
    ("C51", 402, -12.222475, -76.940118),
    ("C53", 402, -12.228781, -76.929727),
    ("C55", 327, -12.221239, -76.936561),
    ("C58", 465, -12.22224, -76.930432),
    ("C61", 402, -12.225766, -76.936258),
    ("C62", 265, -12.22418, -76.940826),
    ("C66", 201, -12.230996, -76.932534),
    ("C116", 263, -12.228592, -76.938892),
    ("C124", 201, -12.230184, -76.936222),
    ("C147", 263, -12.226434, -76.93971),
    ("C150", 201, -12.226503, -76.939203),
]

CD3_LAT = -12.227590
CD3_LON = -76.935571

# nodos
nodes = [{"id": "CD3", "q": 0, "lat": CD3_LAT, "lon": CD3_LON}]
for r in rows:
    nodes.append({"id": r[0], "q": r[1], "lat": r[2], "lon": r[3]})

N = len(nodes)
print("N nodos (incluyendo CD):", N)

# ---------------------------
# Matriz de distancia y tiempo (distancia euclidiana para OR-Tools)
# ---------------------------
dist_matrix = []
time_matrix = []
for i in range(N):
    row_dist = []
    row_time = []
    for j in range(N):
        if i == j:
            row_dist.append(0)
            row_time.append(0)
        else:
            dx = (nodes[i]["lat"] - nodes[j]["lat"]) * 111000  # lat ~ metros
            dy = (nodes[i]["lon"] - nodes[j]["lon"]) * 111000
            d = math.sqrt(dx**2 + dy**2)
            row_dist.append(int(d))
            t = d / (SPEED_KMH * 1000 / 3600)  # segundos
            row_time.append(int(t + SERVICE_SEC))  # incluye servicio
    dist_matrix.append(row_dist)
    time_matrix.append(row_time)

# ---------------------------
# OR-Tools VRP
# ---------------------------
manager = pywrapcp.RoutingIndexManager(N, N, 0)  # N vehículos = N nodos
routing = pywrapcp.RoutingModel(manager)

def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return dist_matrix[from_node][to_node]

transit_callback_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# capacidad
def demand_callback(from_index):
    from_node = manager.IndexToNode(from_index)
    return nodes[from_node]["q"]

demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, [Q]*N, True, "Capacity")

# tiempo
def time_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return time_matrix[from_node][to_node]

time_callback_index = routing.RegisterTransitCallback(time_callback)
routing.AddDimension(
    time_callback_index,
    0,
    3600*24,  # max tiempo por vehículo 24h
    True,
    "Time"
)

# Solver
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
search_parameters.time_limit.seconds = 1800

solution = routing.SolveWithParameters(search_parameters)

# ---------------------------
# Reconstruir rutas
# ---------------------------
routes = []
if solution:
    for vehicle_id in range(N):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        if len(route) > 2:  # descartar rutas vacías
            routes.append(route)
else:
    print("No se encontró solución")

# ---------------------------
# Mapa con rutas reales ORS
# ---------------------------
m = folium.Map(location=[CD3_LAT, CD3_LON], zoom_start=14)
folium.Marker([CD3_LAT, CD3_LON], popup="CD3", icon=folium.Icon(color="red", icon="truck")).add_to(m)
for n in nodes[1:]:
    folium.Marker([n["lat"], n["lon"]], popup=f"{n['id']} (q={n['q']})", icon=folium.Icon(color="blue")).add_to(m)

colors = ["green","purple","orange","brown","gray","olive","cyan","magenta","black"]

for r_idx, route in enumerate(routes):
    coords = [[nodes[i]["lon"], nodes[i]["lat"]] for i in route]
    try:
        rr = requests.post(
            f"https://api.openrouteservice.org/v2/directions/driving-car/geojson",
            json={"coordinates": coords},
            headers={"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
        )
        if rr.status_code == 200:
            gj = rr.json()
            for feat in gj.get("features", []):
                geom = feat.get("geometry", {})
                if geom.get("type") == "LineString":
                    latlon = [[c[1], c[0]] for c in geom["coordinates"]]
                    folium.PolyLine(latlon, color=colors[r_idx % len(colors)], weight=4, opacity=0.7, popup=f"Viaje {r_idx+1}").add_to(m)
        else:
            latlons = [[nodes[i]["lat"], nodes[i]["lon"]] for i in route]
            folium.PolyLine(latlons, color=colors[r_idx % len(colors)], weight=4, opacity=0.6, popup=f"Viaje {r_idx+1}").add_to(m)
    except Exception as e:
        latlons = [[nodes[i]["lat"], nodes[i]["lon"]] for i in route]
        folium.PolyLine(latlons, color=colors[r_idx % len(colors)], weight=4, opacity=0.6, popup=f"Viaje {r_idx+1}").add_to(m)

m.save("mapa_CD3_ortools_RUTAVERDADERA.html")
print("Mapa guardado en: mapa_CD3_ortools.html")

# ---------------------------
# Imprimir rutas y estadísticas
# ---------------------------
total_dist = 0
total_carga = 0
total_time = 0

for r_idx, route in enumerate(routes, start=1):
    route_dist = 0
    route_carga = 0
    route_time = 0
    print(f"Ruta vehículo {r_idx}: ", [nodes[i]["id"] for i in route])
    for pos in range(len(route)-1):
        i = route[pos]
        j = route[pos+1]
        route_dist += dist_matrix[i][j]
        route_time += time_matrix[i][j]
        if j != 0:
            route_carga += nodes[j]["q"]
    print(f"  Carga: {route_carga} kg, Distancia: {route_dist} m, Tiempo aprox: {int(route_time/60)} min")
    total_dist += route_dist
    total_carga += route_carga
    total_time += route_time

print("\nTotal entregado:", total_carga, "kg")
print("Distancia total:", total_dist, "m")
print("Tiempo total aprox:", int(total_time/60), "min")
