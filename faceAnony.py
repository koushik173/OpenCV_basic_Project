import mediapipe as mp
import cv2
import os
import argparse

mp_face_detection = mp.solutions.face_detection

output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

args = argparse.ArgumentParser()

# for live
# args.add_argument("--mode", default='webcam')
# args.add_argument("--filePath", default=None)

# for images
# args.add_argument("--mode", default='images')
# args.add_argument("--filePath", default='./images/hridoi.JPG')

# for recorded video
args.add_argument("--mode", default='video')
args.add_argument("--filePath", default='./vdo/class.mp4')

args = args.parse_args()


# faceDetection
def process_img(img, face_detection):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    H, W, _ = img.shape 
    # print(out.detections)
    
    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)
            
            # print(x1, y1, w, h)
            # img = cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (255,0,0), 5)
            if y1 >= 0 and y1+h <= H and x1 >= 0 and x1+w <= W:
                img[y1:y1+h, x1:x1+w, :] = cv2.blur(img[y1:y1+h, x1:x1+w, :], (30, 30))

            
            # Calculate the center and radius for the circle
            # ix = x1 + w // 2
            # iy = y1 + h // 2
            # r = min(w, h)
            # img = cv2.circle(img, center=(ix, iy), radius=r, color=(0,0,255), thickness=20)

    return img
    


with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    
   if args.mode in ["images"]:
        img = cv2.imread(args.filePath)
        img = process_img(img, face_detection)
        cv2.imshow("Images", img)
        cv2.waitKey(0)
        cv2.imwrite(os.path.join(output_dir, 'output.png'), img)
        
   elif args.mode in ['video']:

        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.avi'),fourcc,25,(frame.shape[1], frame.shape[0]))

        while True:
            ret, frame = cap.read()
            frame = process_img(frame, face_detection)
            output_video.write(frame)
            
            cv2.imshow("webcam",frame)
            if cv2.waitKey(1) & 0xFF == ord('x'):
                 break
            

        cap.release()
        output_video.release()
        
               
   elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(0)
       
        while True:
            ret, frame = cap.read()
            frame = process_img(frame, face_detection)
            
            cv2.imshow("webcam", cv2.flip(frame, 2))
            if cv2.waitKey(1) & 0xFF == ord('x'):
                 break

        cap.release()
        cv2.destroyAllWindows()



       
    
