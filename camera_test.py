import cv2

url = "http://192.168.0.105:8081/video"  # 아이폰 스트림 주소

cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("스트림에서 프레임을 가져올 수 없습니다.")
        break

    cv2.imshow("iPhone IP Camera", frame)

    if cv2.waitKey(1) == 27:  # ESC 키로 종료
        break

cap.release()
cv2.destroyAllWindows()
