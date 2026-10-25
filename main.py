import cv2
video_path = 'minecraft-creeper-pin.mp4'
cascade_path = 'cascade.xml'
output_path ='output6_video.mp4'
object_data = cv2.CascadeClassifier(cascade_path)
video = cv2.VideoCapture(video_path)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'Размер оригинального видео: {frame_width}x{frame_height}')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))
print('Начинаем обработку видео...')
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = object_data.detectMultiScale(gray_frame, minSize=(20,20))
    for (x,y,width,height) in found:
        cv2.rectangle(frame,(x,y),(x + width, y + height), (0,255,0),3)
    out.write(frame)
video.release()