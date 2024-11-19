import cv2

# Load the image
image = cv2.imread(r"C:\Users\urssa\Pictures\IMG-20240816-WA0027[1].jpg")

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# List to store distances and faces
face_distances = []

# Calculate distance of each face from the lower-left corner
for (x, y, w, h) in faces:
    # Calculate the distance from the face center to the lower-left corner
    face_center_x = x + w // 2
    face_center_y = y + h // 2
    distance = ((face_center_x - 0)**2 + (face_center_y - image.shape[0])**2)**0.5
    face_distances.append((distance, (x, y, w, h)))

# Sort faces by distance (ascending order)
face_distances.sort(key=lambda x: x[0])

i=0
for a in face_distances:
    j=0
    for b in a:
        print (f'b[{i}][{j}]: {b}')
        j = j+1
    i = i+1
# Select the two closest faces as targets
target_faces = [face_distances[0][1], face_distances[2][1]] if len(face_distances) > 1 else [face_distances[2][1]]

# Blur all faces except the target faces
for (x, y, w, h) in faces:
    if (x, y, w, h) not in target_faces:  # Skip the target faces
        face_roi = image[y:y+h, x:x+w]  # Region of interest
        blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)  # Apply Gaussian blur
        image[y:y+h, x:x+w] = blurred_face  # Replace the face with blurred version

# Save and display the result
cv2.imwrite("output_image.jpg", image)

