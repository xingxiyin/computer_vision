import cv2
import time
import imutils
from imutils.video import VideoStream

# 初始化摄像头
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    # 从摄像头读取图片
    frame = vs.read()

    # 对图片进行resize
    frame = imutils.resize(frame, width=400)

    #显示摄像头
    cv2.imshow("img", frame)

    #保持画面的持续。
    key = cv2.waitKey(1)
    if key == 27:
        #通过esc键退出摄像
        cv2.destroyAllWindows()
        break
    elif key == ord("s"):
        #通过s键保存图片，并退出。
        cv2.imwrite("image2.jpg", frame)
        cv2.destroyAllWindows()
        break


# Cleanup
cv2.destroyAllWindows()
vs.stop()