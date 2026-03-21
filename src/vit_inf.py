import os
import torch
from accelerate import Accelerator, ProfileKwargs
from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader

class ViTInferencer:
    def __init__(self, data_dir, model_name="google/vit-base-patch16-224-in21k", batch_size=32):
        # Configuración general
        self.data_dir = data_dir
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Configurar ProfileKwargs
        self.profile_kwargs = ProfileKwargs(
            activities=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
            profile_memory=True,
            record_shapes=True,
            schedule_option={"wait": 1, "warmup": 1, "active": 3, "repeat": 0, "skip_first": 1},
            on_trace_ready=self.trace_handler
        )
        
        # Inicializar Accelerate pasando el profiler
        self.accelerator = Accelerator(kwargs_handlers=[self.profile_kwargs])

    def trace_handler(self, p):
        """Manejador que se ejecuta cuando el profiler termina un ciclo activo."""
        print("\n--- Consumo de Memoria ---")
        print(p.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
        # Guarda el archivo en la misma carpeta del script
        p.export_chrome_trace(f"trace_vit_{p.step_num}.json")

    def setup(self):
        """Carga y prepara el modelo, procesador y dataset."""
        print("Preparando modelo y datos para inferencia...")
        
        # Cargar modelo y procesador de imágenes
        self.model = ViTForImageClassification.from_pretrained(self.model_name)
        self.processor = ViTImageProcessor.from_pretrained(self.model_name)
        self.model.eval()

        # Cargar dataset
        self.dataset = load_dataset("imagefolder", data_dir=self.data_dir, split="validation")

        # Crear DataLoader
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            collate_fn=self.collate_fn
        )

        # Preparar componentes con Accelerate
        self.model, self.dataloader = self.accelerator.prepare(
            self.model, self.dataloader
        )

    def collate_fn(self, batch):
        """Transforma las imágenes sobre la marcha."""
        images = [item["image"].convert("RGB") for item in batch]
        inputs = self.processor(images=images, return_tensors="pt")
        return inputs["pixel_values"]

    def infer(self):
        """Ejecuta el bucle de inferencia con el profiler activado."""
        print("Iniciando inferencia con Imagenette local...")
        with self.accelerator.profile() as prof:
            for step, batch in enumerate(self.dataloader):
                with torch.no_grad():
                    outputs = self.model(pixel_values=batch)
                prof.step()
        print("Inferencia finalizada.")

# Bloque de ejecución local (solo se ejecuta si llamas a este archivo directamente)
if __name__ == "__main__":
    inferencer = ViTInferencer(data_dir="../data/img/imagenette2-320", batch_size=32)
    inferencer.setup()
    inferencer.infer()