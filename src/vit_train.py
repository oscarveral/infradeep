import time
import torch
from accelerate import Accelerator
from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader

# Inicializar Accelerate
accelerator = Accelerator()

# Cargar modelo, procesador y optimizador
model_name = "google/vit-base-patch16-224-in21k"
# Especificamos num_labels=10 (las clases de Imagenette) e ignoramos los pesos incompatibles
model = ViTForImageClassification.from_pretrained(model_name, num_labels=10, ignore_mismatched_sizes=True)
processor = ViTImageProcessor.from_pretrained(model_name)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Definir la ruta local y cargar el dataset de entrenamiento
data_dir = "../data/img/imagenette2-320"
dataset = load_dataset("imagefolder", data_dir=data_dir, split="train")

# Función para transformar imágenes y etiquetas sobre la marcha
def collate_fn(batch):
    images = [item["image"].convert("RGB") for item in batch]
    labels = torch.tensor([item["label"] for item in batch])
    inputs = processor(images=images, return_tensors="pt")
    return {"pixel_values": inputs["pixel_values"], "labels": labels}

# Crear DataLoader
training_dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)

# Preparar componentes con Accelerate
model, optimizer, training_dataloader = accelerator.prepare(
    model, optimizer, training_dataloader
)

# Bucle de entrenamiento
model.train()
start_time = time.time()
limite_segundos = 30 * 60 # Límite máximo de 30 minutos para el clúster

print("Iniciando entrenamiento con Imagenette local...")
for epoch in range(1):
    for step, batch in enumerate(training_dataloader):
            
        optimizer.zero_grad()
        
        outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
        loss = outputs.loss
        
        # Retropropagación con Accelerate
        accelerator.backward(loss)
        optimizer.step()
        
        print(f"Paso {step} | Pérdida: {loss.item():.4f}")
        
        # Control de tiempo máximo
        if time.time() - start_time > limite_segundos:
            print("Límite de 30 minutos alcanzado. Deteniendo.")
            break

print("Entrenamiento finalizado.")