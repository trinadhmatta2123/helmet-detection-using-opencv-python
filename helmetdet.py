import cv2

# Load the cascade
helmet_cascade = cv2.CascadeClassifier('helmetdetection.xml')

# Capture video from file or webcam
video = cv2.VideoCapture(0)

while True:
    # Read each frame from the video
    ret, frame = video.read()

    # If the frame could not be grabbed, we've reached the end of the video
    if not ret:
        break

    # Convert the frame into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect helmets
    helmets = helmet_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around the detected helmets
    for (x, y, w, h) in helmets:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the output frame
    cv2.imshow('Helmet Detection', frame)

    # Stop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video object and close all windows
video.release()
cv2.destroyAllWindows()
