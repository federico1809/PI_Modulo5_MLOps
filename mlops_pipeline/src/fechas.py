import pandas as pd

df = pd.read_csv("Base_de_datos_monitoring.csv")
df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"])

print("Rango de fechas:")
print("  Inicio:", df["fecha_prestamo"].min())
print("  Fin:   ", df["fecha_prestamo"].max())
print("  Total meses:", df["fecha_prestamo"].dt.to_period("M").nunique())
print()
print("Registros por mes:")
print(df.groupby(df["fecha_prestamo"].dt.to_period("M")).size().to_string())