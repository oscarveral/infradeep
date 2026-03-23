import time
import torch
import csv
from accelerate import Accelerator
from transformers import ViTForImageClassification, ViTImageProcessor, get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader

# Inicializar Accelerate
accelerator = Accelerator()

# Cargar modelo, procesador y optimizador
model_name = "google/vit-base-patch16-224-in21k"
model = ViTForImageClassification.from_pretrained(model_name, num_labels=10, ignore_mismatched_sizes=True)
processor = ViTImageProcessor.from_pretrained(model_name)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Cargar el dataset de entrenamiento y validación
data_dir = "../data/raw/imagenette2-320"
train_dataset = load_dataset("imagefolder", data_dir=data_dir, split="train")
val_dataset = load_dataset("imagefolder", data_dir=data_dir, split="validation")

def collate_fn(batch):
    images = [item["image"].convert("RGB") for item in batch]
    labels = torch.tensor([item["label"] for item in batch])
    inputs = processor(images=images, return_tensors="pt")
    return {"pixel_values": inputs["pixel_values"], "labels": labels}

train_dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

# Añadir un Scheduler para que el Learning Rate varíe durante el entrenamiento
num_training_steps = len(train_dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# 5. Preparar componentes con Accelerate
model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler
)

# Estructuras para guardar los datos
step_metrics = []
epoch_metrics = []

model.train()
start_time = time.time()
limite_segundos = 30 * 60 # Límite de 30 minutos

print("Iniciando entrenamiento con registro de métricas...")
for epoch in range(1): # Aumenta este número si quieres entrenar más épocas
    total_train_loss = 0
    model.train()

    # BUCLE DE ENTRENAMIENTO (Métricas por paso)
    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        
        outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
        loss = outputs.loss
        
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        
        # Guardar la pérdida y el learning rate actual
        current_lr = optimizer.param_groups[0]['lr']
        step_metrics.append([step, loss.item(), current_lr])
        total_train_loss += loss.item()
        
        if time.time() - start_time > limite_segundos:
            print("Límite de 30 minutos alcanzado. Deteniendo.")
            break

    # BUCLE DE VALIDACIÓN (Métricas por época)
    model.eval()
    total_val_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for batch in val_dataloader:
        with torch.no_grad():
            outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
            total_val_loss += outputs.loss.item()
            
            # Calcular precisión (Accuracy)
            predictions = outputs.logits.argmax(dim=-1)
            correct_predictions += (predictions == batch["labels"]).sum().item()
            total_predictions += batch["labels"].size(0)

    # Calcular las medias finales de la época
    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_val_loss = total_val_loss / len(val_dataloader)
    accuracy = correct_predictions / total_predictions
    
    epoch_metrics.append([epoch + 1, avg_train_loss, avg_val_loss, accuracy])
    print(f"Época {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.4f}")

# Exportar los datos a archivos CSV para poder graficarlos después
with open("metrics_step.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Step", "Train_Loss", "Learning_Rate"])
    writer.writerows(step_metrics)

with open("metrics_epoch.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "Accuracy"])
    writer.writerows(epoch_metrics)

print("Entrenamiento finalizado. Métricas guardadas en CSV.")