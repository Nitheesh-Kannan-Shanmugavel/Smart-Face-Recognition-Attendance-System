## 🚀 How It Works

The Smart Attendance Review System uses **Face Recognition** to automate student attendance from group photos.

### 🧠 Step-by-Step Process:

1. **Load Student Data**
   - Reads reference images of students (stored in folders with their name and roll number).
   - Extracts facial encodings using the `face_recognition` library.

2. **Analyze Group Photo**
   - Loads a classroom group photo.
   - Checks image quality: blur detection and low light enhancement using OpenCV.
   - Enhances and sharpens images to improve recognition accuracy.

3. **Face Detection & Matching**
   - Detects faces in the group photo.
   - Matches each detected face with the known student encodings.
   - Marks identified students as "Present" and mark as "Absent" to students who is not present in image.

4. **Generate Attendance Report**
   - Creates a structured Excel file with roll numbers, names, and presence status.
   - Includes attendance summary (number present, percentage, etc.).

5. **Email the Report**
   - Automatically sends the attendance Excel file to a list of predefined email addresses.

### ✅ Key Features
- Blur and brightness check for image quality.
- Image enhancement for better accuracy.
- Excel report generation.
- Automated email sending.
