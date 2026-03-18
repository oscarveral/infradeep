import os
import time
import csv
import torch
from accelerate import Accelerator, ProfileKwargs
from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader

# Configuración de parámetros para el experimento
# Puedes ajustar el batch size para ver cómo afecta al rendimiento y al consumo de memoria
BATCH_SIZE = 32  

def trace_handler(p):
    print("\n--- Consumo de Memoria por Kernel ---")
    print(p.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    p.export_chrome_trace(f"trace_vit_inf_bs{BATCH_SIZE}_{p.step_num}.json")

profile_kwargs = ProfileKwargs(
    activities=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
    profile_memory=True,
    record_shapes=True,
    schedule_option={"wait": 1, "warmup": 1, "active": 3, "repeat": 0, "skip_first": 1},
    on_trace_ready=trace_handler
)

accelerator = Accelerator(kwargs_handlers=[profile_kwargs])

# Cargar modelo y procesador
model_name = "google/vit-base-patch16-224-in21k"
model = ViTForImageClassification.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)
model.eval()

# Cargar dataset local
data_dir = "../data/raw/imagenette2-320"
dataset = load_dataset("imagefolder", data_dir=data_dir, split="validation")

def collate_fn(batch):
    images = [item["image"].convert("RGB") for item in batch]
    inputs = processor(images=images, return_tensors="pt")
    return inputs["pixel_values"]

val_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# Preparar componentes
model, val_dataloader = accelerator.prepare(model, val_dataloader)

print(f"Iniciando inferencia (Batch Size: {BATCH_SIZE}, Dispositivo: {accelerator.device})...")

# Variables para medir rendimiento
total_images = len(dataset)
start_time = time.time()

with accelerator.profile() as prof:
    for step, batch in enumerate(val_dataloader):
        with torch.no_grad():
            outputs = model(pixel_values=batch)
        prof.step()

# Calcular métricas globales
end_time = time.time()
total_time = end_time - start_time
throughput = total_images / total_time  # Imágenes procesadas por segundo

print(f"\n--- Resumen de Rendimiento ---")
print(f"Tiempo total: {total_time:.2f} segundos")
print(f"Throughput: {throughput:.2f} imágenes/segundo")

# Guardar resultados en un CSV anexando los datos
csv_file = "inference_metrics.csv"
file_exists = os.path.isfile(csv_file)

with open(csv_file, mode="a", newline="") as file:
    writer = csv.writer(file)
    # Escribir cabecera solo si el archivo es nuevo
    if not file_exists:
        writer.writerow(["Device", "Batch_Size", "Total_Images", "Total_Time_sec", "Throughput_img_per_sec"])
    
    # Escribir los datos de esta ejecución
    device_name = "GPU" if torch.cuda.is_available() else "CPU"
    writer.writerow([device_name, BATCH_SIZE, total_images, round(total_time, 2), round(throughput, 2)])

print(f"Métricas agregadas a {csv_file}")