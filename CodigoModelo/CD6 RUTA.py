# vrp_cd6_ortools_map.py
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
# DATOS (CD6 + comedores)
# ---------------------------
# (id, demanda_kg, lat, lon)  -> solo comedores asignados a CD6
rows = [
    ("C28", 340, -12.199226, -76.940103),
    ("C36", 327, -12.208928, -76.944509),
    ("C52", 265, -12.207178, -76.944694),
    ("C67", 340, -12.200539, -76.94813),
    ("C72", 327, -12.198628, -76.948171),
    ("C75", 329, -12.197312, -76.949417),
    ("C76", 201, -12.20185, -76.944234),
    ("C77", 402, -12.196913, -76.942371),
    ("C79", 465, -12.202817, -76.944847),
    ("C80", 263, -12.194237, -76.946965),
    ("C82", 327, -12.194909, -76.953116),
    ("C84", 327, -12.19737, -76.945139),
    ("C86", 265, -12.200504, -76.954925),
    ("C94", 465, -12.195595, -76.945289),
    ("C98", 263, -12.20465, -76.948198),
    ("C99", 464, -12.194116, -76.94212),
    ("C101",464, -12.197448, -76.946484),
    ("C104",201, -12.199496, -76.947684),
    ("C112",402, -12.19571, -76.942859),
    ("C115",464, -12.201881, -76.948991),
    ("C119",402, -12.199302, -76.946332),
    ("C122",464, -12.198574, -76.94978),
    ("C127",340, -12.19667,  -76.940292),
    ("C129",326, -12.200592, -76.95012),
    ("C131",402, -12.199669, -76.945944),
    ("C132",465, -12.205945, -76.948215),
    ("C135",263, -12.204472, -76.945002),
    ("C136",402, -12.197246, -76.951405),
    ("C146",327, -12.198363, -76.949146),
]

# Coordenadas del depósito CD6 (promedio de los comedores CD6)
CD6_LAT = -12.199660
CD6_LON = -76.946706

# nodos (depot primero)
nodes = [{"id": "CD6", "q": 0, "lat": CD6_LAT, "lon": CD6_LON}]
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
# Nota: aquí se crea N vehículos (uno por nodo). Ajusta si quieres un número distinto.
manager = pywrapcp.RoutingIndexManager(N, N, 0)  # N vehículos = N nodos, depósito index 0
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
m = folium.Map(location=[CD6_LAT, CD6_LON], zoom_start=14)
folium.Marker([CD6_LAT, CD6_LON], popup="CD6", icon=folium.Icon(color="red", icon="truck")).add_to(m)
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

m.save("mapa_CD6_ortools_RUTAVERDADERA.html")
print("Mapa guardado en: mapa_CD6_ortools_RUTAVERDADERA.html")

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
