import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
from PIL import Image
import numpy as np

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f]

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load('fine_tuned_pokemon_classifier_V2.pth', map_location=device))
model = model.to(device)
model.eval()

# Define transformation for incoming frames
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def predict_pokemon(roi):
    # Convert ROI to PIL image
    img = Image.fromarray(roi)
    
    # Apply transformations
    img = transform(img).unsqueeze(0).to(device)
    
    # Perform prediction
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item() * 100
        pokemon_name = class_names[predicted.item()]
    
    return pokemon_name, confidence

# Open a connection to the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit the real-time Pokémon recognition.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to grayscale and apply GaussianBlur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Detect edges and find contours
    edged = cv2.Canny(blurred, 30, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Filter out small contours
        if cv2.contourArea(contour) < 500:
            continue

        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        roi = frame[y:y+h, x:x+w]

        # Predict Pokémon in the region of interest (ROI)
        pokemon_name, confidence = predict_pokemon(roi)

        # Draw bounding box and display prediction if confidence is high enough
        if confidence > 90:  # Adjust confidence threshold as needed
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{pokemon_name}: {confidence:.2f}%"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the frame with predictions
    cv2.imshow('Real-Time Pokémon Recognition', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
