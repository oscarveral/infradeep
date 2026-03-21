import time
import torch
from accelerate import Accelerator
from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader

class ViTTrainer:
    def __init__(self, data_dir, model_name="google/vit-base-patch16-224-in21k", batch_size=32, lr=5e-5, time_limit_mins=30):
        # Configuración general
        self.data_dir = data_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.lr = lr
        self.time_limit_secs = time_limit_mins * 60
        
        # Inicializar Accelerate
        self.accelerator = Accelerator()
        
    def setup(self):
        """Carga y prepara el modelo, procesador, dataset y el optimizador."""
        print("Preparando modelo y datos...")
        
        # Cargar modelo y procesador
        self.model = ViTForImageClassification.from_pretrained(
            self.model_name, num_labels=10, ignore_mismatched_sizes=True
        )
        self.processor = ViTImageProcessor.from_pretrained(self.model_name)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        
        # Cargar dataset
        self.dataset = load_dataset("imagefolder", data_dir=self.data_dir, split="train")
        
        # Crear DataLoader
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            collate_fn=self.collate_fn, 
            shuffle=True
        )
        
        # Preparar componentes con Accelerate
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )

    def collate_fn(self, batch):
        """Transforma las imágenes y etiquetas sobre la marcha."""
        images = [item["image"].convert("RGB") for item in batch]
        labels = torch.tensor([item["label"] for item in batch])
        inputs = self.processor(images=images, return_tensors="pt")
        return {"pixel_values": inputs["pixel_values"], "labels": labels}

    def train(self, epochs=1):
        """Ejecuta el bucle de entrenamiento respetando el límite de tiempo."""
        self.model.train()
        start_time = time.time()
        
        print("Iniciando entrenamiento con Imagenette local...")
        for epoch in range(epochs):
            for step, batch in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                
                outputs = self.model(pixel_values=batch["pixel_values"], labels=batch["labels"])
                loss = outputs.loss
                
                # Retropropagación con Accelerate
                self.accelerator.backward(loss)
                self.optimizer.step()
                
                print(f"Época {epoch+1} | Paso {step} | Pérdida: {loss.item():.4f}")
                
                # Control de tiempo máximo
                if time.time() - start_time > self.time_limit_secs:
                    print("Límite de tiempo alcanzado. Deteniendo.")
                    return  # Termina la ejecución de la función por completo

        print("Entrenamiento finalizado.")

# Bloque de ejecución local (solo se ejecuta si llamas a este archivo directamente)
if __name__ == "__main__":
    trainer = ViTTrainer(data_dir="../data/img/imagenette2-320", batch_size=32)
    trainer.setup()
    trainer.train(epochs=1)