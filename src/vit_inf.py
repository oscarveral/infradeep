import os
import torch
from accelerate import Accelerator, ProfileKwargs
from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader

def trace_handler(p):
    print("\n--- Consumo de Memoria ---")
    print(p.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    # Guarda el archivo en la misma carpeta del script para evitar errores de rutas en Windows
    p.export_chrome_trace(f"trace_vit_{p.step_num}.json")

profile_kwargs = ProfileKwargs(
    activities=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
    profile_memory=True,
    record_shapes=True,
    schedule_option={"wait": 1, "warmup": 1, "active": 3, "repeat": 0, "skip_first": 1},
    on_trace_ready=trace_handler
)

accelerator = Accelerator(kwargs_handlers=[profile_kwargs])

# Cargar modelo y procesador de imágenes
model_name = "google/vit-base-patch16-224-in21k"
model = ViTForImageClassification.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)
model.eval()

# Definir la ruta relativa desde la carpeta src/ a la carpeta raíz de las imágenes
data_dir = "../data/img/imagenette2-320"

# Cargar el dataset usando 'imagefolder'. Automáticamente detectará las carpetas 'train' y 'val'
dataset = load_dataset("imagefolder", data_dir=data_dir, split="validation")

# Función para transformar las imágenes sobre la marcha
def collate_fn(batch):
    images = [item["image"].convert("RGB") for item in batch]
    inputs = processor(images=images, return_tensors="pt")
    return inputs["pixel_values"]

# Crear DataLoader
val_dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# Preparar componentes con Accelerate
model, val_dataloader = accelerator.prepare(model, val_dataloader)

print("Iniciando inferencia con Imagenette local...")
with accelerator.profile() as prof:
    for step, batch in enumerate(val_dataloader):
        with torch.no_grad():
            outputs = model(pixel_values=batch)
        prof.step()
print("Inferencia finalizada.")