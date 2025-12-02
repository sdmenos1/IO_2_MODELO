import pandas as pd
import openrouteservice
import time
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, value
import folium

# --------------------------------------------------------------------
# 1. PREPARACIÓN INICIAL (API KEY, DATOS)
# --------------------------------------------------------------------
API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjhhOWE5NmYxMzZiNjQ5OWI5MTgxMDEyMTUzMDJhYjk4IiwiaCI6Im11cm11cjY0In0="
client = openrouteservice.Client(key=API_KEY)

# DATOS: 8 Centros de Distribución (CD)
cd_data = {
    "CD": ["CD1", "CD2", "CD3", "CD4", "CD5", "CD6", "CD7", "CD8"],
    "lat": [-12.216351, -12.207688, -12.227590, -12.189898, -12.208121, -12.199713, -12.233326, -12.239124],
    "lon": [-76.950706, -76.936671, -76.935571, -76.951337, -76.953076, -76.946786, -76.938849, -76.928217],
    "capacidad": [9042, 6036, 11411, 6978, 3571, 10541, 12613, 19599]
}
cds = pd.DataFrame(cd_data)

# DATOS: 150 COMEDORES
comedores_data = {
    "Comedor": [
        "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10",
        "C11","C12","C13","C14","C15","C16","C17","C18","C19","C20",
        "C21","C22","C23","C24","C25","C26","C27","C28","C29","C30",
        "C31","C32","C33","C34","C35","C36","C37","C38","C39","C40",
        "C41","C42","C43","C44","C45","C46","C47","C48","C49","C50",
        "C51","C52","C53","C54","C55","C56","C57","C58","C59","C60",
        "C61","C62","C63","C64","C65","C66","C67","C68","C69","C70",
        "C71","C72","C73","C74","C75","C76","C77","C78","C79","C80",
        "C81","C82","C83","C84","C85","C86","C87","C88","C89","C90",
        "C91","C92","C93","C94","C95","C96","C97","C98","C99","C100",
        "C101","C102","C103","C104","C105","C106","C107","C108","C109","C110",
        "C111","C112","C113","C114","C115","C116","C117","C118","C119","C120",
        "C121","C122","C123","C124","C125","C126","C127","C128","C129","C130",
        "C131","C132","C133","C134","C135","C136","C137","C138","C139","C140",
        "C141","C142","C143","C144","C145","C146","C147","C148","C149","C150"
    ],
    "lat": [
        -12.228684, -12.215963, -12.207320, -12.220941, -12.220380, -12.216099, -12.224928, -12.221842, -12.213511, -12.223859,
        -12.224404, -12.210318, -12.221413, -12.227425, -12.224324, -12.222502, -12.221013, -12.227254, -12.217377, -12.229279,
        -12.218914, -12.215616, -12.217434, -12.209152, -12.206931, -12.210953, -12.224027, -12.199226, -12.212747, -12.220861,
        -12.216103, -12.204266, -12.207971, -12.222860, -12.211689, -12.208928, -12.227225, -12.222848, -12.209522, -12.204909,
        -12.225716, -12.219265, -12.210086, -12.202893, -12.225223, -12.218641, -12.208532, -12.224881, -12.217875, -12.203925,
        -12.222475, -12.207178, -12.228781, -12.216210, -12.221239, -12.207671, -12.223930, -12.222240, -12.209973, -12.216895,
        -12.225766, -12.224180, -12.203829, -12.201579, -12.187621, -12.230996, -12.200539, -12.190014, -12.238410, -12.189841,
        -12.190125, -12.198628, -12.235437, -12.234152, -12.197312, -12.201850, -12.196913, -12.238942, -12.202817, -12.194237,
        -12.235258, -12.194909, -12.233129, -12.197370, -12.191450, -12.200504, -12.225724, -12.226867, -12.203056, -12.227396,
        -12.187344, -12.230167, -12.190354, -12.195595, -12.239422, -12.235609, -12.238262, -12.204650, -12.194116, -12.236743,
        -12.197448, -12.191919, -12.232100, -12.199496, -12.186543, -12.186777, -12.236453, -12.230456, -12.236650, -12.191502,
        -12.192858, -12.195710, -12.187764, -12.198312, -12.201881, -12.228592, -12.233179, -12.188455, -12.199302, -12.186005,
        -12.208202, -12.198574, -12.188438, -12.230184, -12.213812, -12.232534, -12.196670, -12.237475, -12.200592, -12.206564,
        -12.199669, -12.205945, -12.187638, -12.235140, -12.204472, -12.197246, -12.228851, -12.189818, -12.190553, -12.203154,
        -12.227103, -12.233834, -12.225433, -12.229842, -12.189872, -12.198363, -12.226434, -12.229651, -12.192577, -12.226503
    ],
    "lon": [
        -76.946888, -76.943237, -76.941006, -76.934228, -76.930412, -76.928893, -76.929422, -76.942382, -76.941365, -76.932730,
        -76.928796, -76.934330, -76.946707, -76.935243, -76.937127, -76.931024, -76.937749, -76.933929, -76.930534, -76.931166,
        -76.940158, -76.934469, -76.928081, -76.954383, -76.935798, -76.941605, -76.935597, -76.940103, -76.942960, -76.944927,
        -76.932664, -76.936825, -76.955091, -76.933621, -76.949953, -76.944509, -76.935647, -76.943664, -76.955152, -76.939234,
        -76.936778, -76.938152, -76.946441, -76.938586, -76.942540, -76.938020, -76.937521, -76.935366, -76.952046, -76.940315,
        -76.940118, -76.944694, -76.929727, -76.950334, -76.936561, -76.934920, -76.943776, -76.930432, -76.947087, -76.950781,
        -76.936258, -76.940826, -76.941482, -76.954474, -76.953027, -76.932534, -76.948130, -76.953278, -76.936864, -76.951293,
        -76.946186, -76.948171, -76.943783, -76.937544, -76.949417, -76.944234, -76.942371, -76.936457, -76.944847, -76.946965,
        -76.933109, -76.953116, -76.943475, -76.945139, -76.953361, -76.954925, -76.947431, -76.948316, -76.954142, -76.939568,
        -76.950333, -76.941201, -76.947089, -76.945289, -76.933756, -76.936261, -76.935386, -76.948198, -76.942120, -76.939965,
        -76.946484, -76.950556, -76.949524, -76.947684, -76.947946, -76.949135, -76.934578, -76.943312, -76.942244, -76.955888,
        -76.945235, -76.942859, -76.942859, -76.956683, -76.948991, -76.938892, -76.938892, -76.944263, -76.946332, -76.951421,
        -76.953203, -76.949780, -76.950117, -76.936222, -76.952343, -76.947539, -76.940292, -76.930448, -76.950120, -76.951343,
        -76.945944, -76.948215, -76.953293, -76.933393, -76.945002, -76.951405, -76.940547, -76.948092, -76.948687, -76.952724,
        -76.948787, -76.947255, -76.944694, -76.939933, -76.941287, -76.949146, -76.939710, -76.946397, -76.949849, -76.939203
    ],
    "kg": [
        201,465,402,327,465,263,402,340,328,201, 326,464,465,263,201,464,465,201,263,265,
        201,465,326,340,402,265,465,340,263,265, 327,465,340,327,264,327,465,465,327,327,
        340,327,201,340,327,265,465,327,465,201, 402,265,402,402,327,265,327,465,465,201,
        402,265,402,465,201,201,340,465,327,465, 402,327,263,263,329,201,402,264,465,263,
        201,327,328,327,265,265,465,201,465,327, 328,263,201,465,265,327,327,263,464,265,
        464,465,265,201,465,340,340,327,327,201, 265,402,327,327,464,263,263,327,402,465,
        327,464,263,201,465,402,340,265,326,327, 402,465,402,263,263,402,327,201,201,327,
        340,327,265,327,340,327,263,340,201,201
    ]
}
comedores = pd.DataFrame(comedores_data)

# --------------------------------------------------------------------
# 2. FUNCIÓN DE DISTANCIA Y CÁLCULO DE MATRIZ DE COSTOS (D_ij)
# --------------------------------------------------------------------
def distancia_ors(lon1, lat1, lon2, lat2):
    try:
        coords = ((lon1, lat1), (lon2, lat2))
        ruta = client.directions(coords) 
        distancia = ruta["routes"][0]["summary"]["distance"]
        return distancia
    except Exception as e:
        print(f"Error en ORS para {lat1},{lon1} a {lat2},{lon2}: {e}")
        return 1e12 

D_ij = {}
total_calls = len(comedores) * len(cds)
print(f"Calculando matriz de distancias (Comedores x CD). Total {total_calls} llamadas...")
print(f"Tiempo estimado: aproximadamente {total_calls * 1.2 / 60:.1f} minutos.")

for i, com in comedores.iterrows():
    comedor_id = com["Comedor"]
    D_ij[comedor_id] = {}
    
    for j, cd in cds.iterrows():
        cd_id = cd["CD"]
        dist = distancia_ors(com["lon"], com["lat"], cd["lon"], cd["lat"])
        D_ij[comedor_id][cd_id] = dist
        time.sleep(1.2)

print("Cálculo de distancias completado.")

# --------------------------------------------------------------------
# 3. MODELO DE PROGRAMACIÓN LINEAL ENTERA (PLE) CON PULP
# --------------------------------------------------------------------
Comedores = comedores['Comedor'].tolist()
CDs = cds['CD'].tolist()
K_i = comedores.set_index("Comedor")["kg"].to_dict()
C_j = cds.set_index("CD")["capacidad"].to_dict()

modelo = LpProblem("Asignacion_Optima_Comedores", LpMinimize)
X_ij = LpVariable.dicts("Asignacion", (Comedores, CDs), 0, 1, 'Binary')

modelo += lpSum(D_ij[i][j] * X_ij[i][j] for i in Comedores for j in CDs), "Distancia_Total_Minima"

for i in Comedores:
    modelo += lpSum(X_ij[i][j] for j in CDs) == 1, f"Comedor_{i}_Asignacion_Unica"

for j in CDs:
    modelo += lpSum(K_i[i] * X_ij[i][j] for i in Comedores) <= C_j[j], f"CD_{j}_Restriccion_Capacidad"

print("\nResolviendo el modelo de Optimización Óptima...")
modelo.solve()

# --------------------------------------------------------------------
# 4. EXTRACCIÓN DE RESULTADOS
# --------------------------------------------------------------------
asignaciones_optimas = []
for i in Comedores:
    for j in CDs:
        if value(X_ij[i][j]) == 1.0:
            asignaciones_optimas.append({
                "Comedor": i,
                "CD_asignado": j,
                "kg": K_i[i],
                "distancia_metros": D_ij[i][j]
            })

resultado_optimo = pd.DataFrame(asignaciones_optimas)
resultado_optimo = resultado_optimo.merge(comedores[['Comedor', 'lat', 'lon']], on='Comedor')

resultado_optimo['distancia_km'] = resultado_optimo['distancia_metros'] / 1000
km_total_por_cd = resultado_optimo.groupby("CD_asignado")["distancia_km"].sum().round(2)

print("\n=== DISTANCIA TOTAL ASIGNADA POR CD (en Kilómetros) ===")
print(km_total_por_cd.to_string())

ruta_excel_optimo = "resultado_asignacion_optima_150.xlsx"
resultado_optimo.to_excel(ruta_excel_optimo, index=False)
print(f"\nArchivo Excel generado exitosamente: {ruta_excel_optimo}")

# --------------------------------------------------------------------
# 5. MAPA DE RESULTADOS ÓPTIMOS (ESTILO MEJORADO)
# --------------------------------------------------------------------

lat_centro = comedores["lat"].mean()
lon_centro = comedores["lon"].mean()

# 1. Base del mapa más limpia con "CartoDB positron"
mapa_optimo = folium.Map(location=[lat_centro, lon_centro], zoom_start=13, tiles="CartoDB positron")

# 2. Definición de Colores (para 8 CDs)
colores = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue", "black"]
color_por_cd = {cd: colores[i % len(colores)] for i, cd in enumerate(CDs)}

# 3. Dibujar CDs (Marcadores Grandes tipo Almacén)
for _, cd in cds.iterrows():
    folium.Marker(
        location=[cd["lat"], cd["lon"]],
        popup=f"CD: {cd['CD']}<br>Capacidad: {cd['capacidad']}",
        icon=folium.Icon(color=color_por_cd[cd['CD']], icon="warehouse", prefix="fa")
    ).add_to(mapa_optimo)

# 4. Dibujar Comedores (Círculos Pequeños) y Líneas Conectoras
for _, row in resultado_optimo.iterrows():
    cd_asignado = row["CD_asignado"]
    color = color_por_cd.get(cd_asignado, "gray")

    # Marcador de Comedor (Círculo pequeño para no saturar)
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=f"Comedor: {row['Comedor']}<br>kg: {row['kg']}<br>Asignado a: {cd_asignado}<br>Dist: {row['distancia_km']:.2f} km"
    ).add_to(mapa_optimo)

    # Línea conectora
    cd_row = cds[cds["CD"] == cd_asignado].iloc[0]
    folium.PolyLine(
        locations=[
            [cd_row["lat"], cd_row["lon"]],
            [row["lat"], row["lon"]]
        ],
        color=color,
        weight=1,      # Línea más fina
        opacity=0.6    # Línea semi-transparente
    ).add_to(mapa_optimo)

# Guardar mapa
mapa_optimo.save("mapa_asignacion_optima_estilo_mejorado.html")
print("\nMapa óptimo generado exitosamente: mapa_asignacion_optima_estilo_mejorado.html")