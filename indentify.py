import face_recognition
from PIL import Image, ImageDraw

image_of_ganesh = face_recognition.load_image_file('./img/ganesh.jpg')
ganesh_face_encoding = face_recognition.face_encodings(image_of_ganesh)[0]

image_of_akhil = face_recognition.load_image_file('./img/akhil.jpg')
akhil_face_encoding = face_recognition.face_encodings(image_of_akhil)[0]

image_of_sravan = face_recognition.load_image_file("img/sravan.jpg")
sravan_face_encoding = face_recognition.face_encodings(image_of_sravan)[0]

image_of_jicku = face_recognition.load_image_file("img/jicku.jpg")
jicku_face_encoding = face_recognition.face_encodings(image_of_jicku)[0]

image_of_abin = face_recognition.load_image_file("img/abin.jpg")
abin_face_encoding = face_recognition.face_encodings(image_of_abin)[0]


#  Create arrays of encodings and names
known_face_encodings = [
  ganesh_face_encoding,
  akhil_face_encoding,
  sravan_face_encoding,
  jicku_face_encoding,
  abin_face_encoding
]

known_face_names = [
  "Ganesh",
  "Akhil",
  "Sravan",
  "Jicku",
  "Abin"
]

# Load test image to find faces in
test_image = face_recognition.load_image_file("img/")

# Find faces in test image
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

# Convert to PIL format
pil_image = Image.fromarray(test_image)

# Create a ImageDraw instance
draw = ImageDraw.Draw(pil_image)

# Loop through faces in test image
for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

  name = "Unknown Person"

  # If match
  if True in matches:
    first_match_index = matches.index(True)
    name = known_face_names[first_match_index]
  
  # Draw box
  draw.rectangle(((left, top), (right, bottom)), outline=(255,255,0))

  # Draw label
  text_width, text_height = draw.textsize(name)
  draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(255,255,0), outline=(255,255,0))
  draw.text((left + 6, bottom - text_height - 5), name, fill=(0,0,0))

del draw

# Display image
pil_image.show()

# Save image
pil_image.save('identify1.jpg')