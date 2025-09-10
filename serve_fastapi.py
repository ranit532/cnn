from fastapi import FastAPI, File, UploadFile
import mlflow
import torch
from PIL import Image
from torchvision import transforms
import io

# Initialize the FastAPI app
app = FastAPI(title="Shapes Classifier API")

# --- Load the MLflow model ---
# NOTE: Replace this with the actual run ID from your MLflow experiment.
RUN_ID = "9be6226dcddd4f7e96b4fd43525078d2"
LOGGED_MODEL = f"runs:/{RUN_ID}/model"

# Load the model
model = mlflow.pytorch.load_model(LOGGED_MODEL)
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define the label map (inverse of the one from training)
label_map = {0: 'circle', 1: 'square', 2: 'triangle'} # Replace with your actual label map if different

@app.get("/")
def read_root():
    return {"message": "Welcome to the Shapes Classifier API. Use the /predict/ endpoint to classify an image."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file and returns the predicted class.
    """
    # Read the image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)

    # Make a prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_idx = torch.max(output, 1)
    
    predicted_label = label_map[predicted_idx.item()]

    return {"prediction": predicted_label}

# To run this app, use the command:
# uvicorn serve_fastapi:app --reload