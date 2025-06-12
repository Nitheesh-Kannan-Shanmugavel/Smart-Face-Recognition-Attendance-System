import os
import face_recognition
import pandas as pd
import datetime
from openpyxl import load_workbook
import cv2
import matplotlib.pyplot as plt
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import threading
import time

PERIODS = {
    "Period 1": ("8:30 AM", "9:20 AM"),
    "Period 2": ("9:20 AM", "10:10 AM"),
    "Period 3": ("10:30 AM", "11:20 AM"),
    "Period 4": ("11:20 AM", "12:10 PM"),
    "Period 5": ("1:40 PM", "2:30 PM"),
    "Period 6": ("2:30 PM", "3:20 PM"),
    "Period 7": ("3:30 PM", "4:20 PM"),
    "Period 8": ("4:20 PM", "5:10 PM"),
}

def close_matplotlib_after_delay(delay=5):
    time.sleep(delay)
    plt.close()

def get_current_period():
    now = datetime.datetime.now().time()
    for period, (start_time, end_time) in PERIODS.items():
        start = datetime.datetime.strptime(start_time, "%I:%M %p").time()
        end = datetime.datetime.strptime(end_time, "%I:%M %p").time()
        if start <= now <= end:
            return period
    return "Others"

def is_image_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"Laplacian Variance: {laplacian_var}")
    return laplacian_var < threshold

def is_image_low_light(image, threshold=50):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    print(f"Average Brightness: {avg_brightness}")
    return avg_brightness < threshold

def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(enhanced_image, -1, kernel)
    return sharpened_image

def enhance_low_light(image):
    gamma = 1.5
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def load_known_faces(students_folder):
    known_face_encodings = []
    known_face_names = []
    known_face_rollnos = []
    for student_folder_name in os.listdir(students_folder):
        student_folder_path = os.path.join(students_folder, student_folder_name)
        if os.path.isdir(student_folder_path):
            try:
                name, roll_number = student_folder_name.split('_')
            except ValueError:
                print(f"Skipping folder with unexpected name format: {student_folder_name}")
                continue
            for image_name in os.listdir(student_folder_path):
                image_path = os.path.join(student_folder_path, image_name)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(name)
                    known_face_rollnos.append(roll_number)
                else:
                    print(f"No face encodings found in image: {image_path}")
    return known_face_encodings, known_face_names, known_face_rollnos

def recognize_students(group_photo_folder, known_face_names, known_face_rollnos, known_face_encodings):
    recognized_students = set()
    for file_name in os.listdir(group_photo_folder):
        if file_name.lower().endswith(('png', 'jpg', 'jpeg')):
            group_photo_path = os.path.join(group_photo_folder, file_name)
            try:
                group_photo = cv2.imread(group_photo_path)
                if group_photo is None:
                    print(f"Error loading group photo {file_name}: File is not a valid image.")
                    continue
                if is_image_blurry(group_photo):
                    print(f"Image {file_name} is too blurry for face detection.")
                    plt.imshow(cv2.cvtColor(group_photo, cv2.COLOR_BGR2RGB))
                    plt.title("Image is too blurry for face detection.")
                    plt.axis('off')
                    plt.show()
                    close_matplotlib_after_delay(5)
                    choice = input("Do you want to upload another photo? (yes/no): ").strip().lower()
                    if choice == 'yes':
                        return recognize_students(group_photo_folder, known_face_names, known_face_rollnos, known_face_encodings)
                    else:
                        print("Process stopped.")
                        return []
                if is_image_low_light(group_photo):
                    print(f"Image {file_name} is too dark. Enhancing image...")
                    enhanced_photo = enhance_low_light(group_photo)
                    if is_image_low_light(enhanced_photo, threshold=50):
                        print(f"Image {file_name} is still too dark after enhancement.")
                        plt.imshow(cv2.cvtColor(enhanced_photo, cv2.COLOR_BGR2RGB))
                        plt.title("Image is too dark for face detection.")
                        plt.axis('off')
                        plt.show()
                        close_matplotlib_after_delay(5)
                        choice = input("Do you want to upload another photo? (yes/no): ").strip().lower()
                        if choice == 'yes':
                            return recognize_students(group_photo_folder, known_face_names, known_face_rollnos, known_face_encodings)
                        else:
                            print("Process stopped.")
                            return []
                    group_photo = enhanced_photo
                enhanced_photo = enhance_image(group_photo)
                group_photo_rgb = cv2.cvtColor(enhanced_photo, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(group_photo_rgb)
                face_encodings = face_recognition.face_encodings(group_photo_rgb, face_locations)
                print(f"Detected {len(face_encodings)} faces in the group photo {file_name}.")
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                    distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    name = "Unknown"
                    roll_number = "Unknown"
                    if True in matches:
                        best_match_index = distances.argmin()
                        name = known_face_names[best_match_index]
                        roll_number = known_face_rollnos[best_match_index]
                    if name != "Unknown" and roll_number != "Unknown":
                        recognized_students.add((name, roll_number))
                        print(f"{name} ({roll_number}) recognized")
                        cv2.rectangle(group_photo_rgb, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(group_photo_rgb, f"{name} ({roll_number})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(group_photo_rgb, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(group_photo_rgb, "Unrecognized", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                plt.imshow(group_photo_rgb)
                plt.title(f"Recognized Faces in {file_name}")
                plt.axis('off')
                plt.show()
                close_matplotlib_after_delay(5)
            except Exception as e:
                print(f"Error processing group photo {file_name}: {e}")
    recognized_students = list(recognized_students)
    return recognized_students

def adjust_column_widths(file_path):
    workbook = load_workbook(file_path)
    worksheet = workbook.active
    for column in worksheet.columns:
        max_length = 0
        column = [cell for cell in column]
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(cell.value)
            except:
                pass
        adjusted_width = (max_length + 2)
        worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
    workbook.save(file_path)

def mark_attendance(recognized_students, attendance_folder, known_face_rollnos, known_face_names):
    current_period = get_current_period()
    attendance_df = pd.DataFrame(columns=['Roll Number', 'Name', current_period])
    recognized_set = {(name, roll_number) for name, roll_number in recognized_students}
    
    for roll_number, name in zip(known_face_rollnos, known_face_names):
        if (name, roll_number) in recognized_set:
            attendance_df = pd.concat([attendance_df, pd.DataFrame([{'Roll Number': roll_number, 'Name': name, current_period: 'Present'}])], ignore_index=True)
        else:
            attendance_df = pd.concat([attendance_df, pd.DataFrame([{'Roll Number': roll_number, 'Name': name, current_period: 'Absent'}])], ignore_index=True)
    
    attendance_df.sort_values(by='Roll Number', inplace=True)
    
    present_count = len(attendance_df[attendance_df[current_period] == 'Present'])
    absent_count = len(attendance_df[attendance_df[current_period] == 'Absent'])
    total_count = len(attendance_df)
    
 
    summary_df = pd.DataFrame({
        'Roll Number': ['Summary'],
        'Name': [''],
        current_period: [f'Present: {present_count} ({present_count/total_count*100:.1f}%), Absent: {absent_count} ({absent_count/total_count*100:.1f}%)']
    })
    

    attendance_df = pd.concat([attendance_df, summary_df], ignore_index=True)
    
    date_time_str = datetime.datetime.now().strftime("Attendance Report EEE (3rd Year) - %d-%m-%Y %H-%M-%S.xlsx")
    attendance_file = os.path.join(attendance_folder, date_time_str)

    with pd.ExcelWriter(attendance_file, engine='openpyxl') as writer:
        attendance_df.to_excel(writer, index=False, sheet_name='Attendance')
        
     
        workbook = writer.book
        worksheet = writer.sheets['Attendance']
        
        last_row = len(attendance_df)
        worksheet[f'C{last_row}'].font = 'bold'
        
  
        for column in worksheet.columns:
            max_length = 0
            column = [cell for cell in column]
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = (max_length + 2)
            worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
    
    print(f"Attendance marked successfully. File saved as {attendance_file}")
    os.startfile(attendance_file)
    send_email(attendance_file)
def send_email(file_path):
    sender_email = "nitheeshkannanshanmugavel@gmail.com"
    receiver_email = ["22e138@psgtech.ac.in", "22e146@psgtech.ac.in", "22e121@psgtech.ac.in", "23e401@psgtech.ac.in", "23e407@psgtech.ac.in"]
    password = "vvop pkfk uqar nysx"
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = ", ".join(receiver_email)  
    msg['Subject'] = "Attendance Report"
    
    with open(file_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(file_path)}")
        msg.attach(part)
    
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")

def main():
    print("Script started.")
    students_folder = 'students'
    attendance_folder = 'attendance_records'
    group_photo_folder = 'group photo'
    if not os.path.exists(attendance_folder):
        os.makedirs(attendance_folder)
    known_face_encodings, known_face_names, known_face_rollnos = load_known_faces(students_folder)
    recognized_students = recognize_students(group_photo_folder, known_face_names, known_face_rollnos, known_face_encodings)
    print("Students recognized:", recognized_students)
    if recognized_students:
        mark_attendance(recognized_students, attendance_folder, known_face_rollnos, known_face_names)
    else:
        print("No students recognized.")

if __name__ == "__main__":
    main()
