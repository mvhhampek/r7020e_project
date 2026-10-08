from utils import *
import cv2
import numpy as np
import cv2
import time
from ultralytics import YOLO
from PIL import Image

def main():
    verbose = True
    print("Loading model. . .")

    model = YOLO("best.pt")

    K, D, R, P, height, width, distortion_model, frame_id = read_camera_info()
    
    print("Loading dataset. . .")
    data = get_matched_data(verbose)
    
    map1, map2 = cv2.initUndistortRectifyMap(K, D, R, P, (640, 400), cv2.CV_16SC2)
    for timestamp in data.keys():
        current = data[timestamp]
        current['color_image'] = cv2.remap(current['color_image'], map1, map2, interpolation=cv2.INTER_LINEAR)
    print("\n\n")

    
    yolo_times = []
    localization_times = []
    make_a_movie_ts_images = []
    idx = 0
    for timestamp in data.keys():

        d = data[timestamp]

        depth_img = d['depth_image']
        color_img = d['color_image']

        t0 = time.perf_counter()
        results = model(color_img, conf=0.4, iou=0.2, verbose=verbose)
        t1 = time.perf_counter()
        yolo_times.append(t1-t0)
        
        annotated_color = color_img.copy()
        annotated_depth = depth_img.copy()
        
        depth_norm = (annotated_depth - annotated_depth.min()) / (annotated_depth.max() - annotated_depth.min() + 1e-6)
        depth_norm = (depth_norm * 255).astype(np.uint8)
        sampled_depth_points = []
        det_times = []
        for box in results[0].boxes:
            
            t0 = time.perf_counter()
            
            box_xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, box_xyxy)

            cropped_depth = depth_img[y1:y2, x1:x2]
            xyz_mm, global_pos = get_detected_object_xyz(depth_img, cropped_depth, P, x1, y1)
            x, y, z = xyz_mm * 1e-3
            
            
            real = classify_detection(cropped_depth)
            if real is None:
                continue
            
            
            t1 = time.perf_counter()
            det_times.append(t1-t0)
            
            sampled_depth_points.append(global_pos)
            
            
            
            bbox_color = (0, 255, 20) if real else (0, 0, 255)
            title = "REAL" if real else "DECOY"
            title += f"   conf: {box.conf[0]:.2f}"
            pos = f"x:{x:.2f}, y:{y:.2f}, z:{z:.2f}"
            lines = [title, pos]

            y0 = -20
            dy = 15
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(pos, font, font_scale, thickness)
            cv2.rectangle(annotated_color, (x1, y1 + y0 - text_h - baseline), (x1 + text_w, y1 + y0 + dy + baseline), (0, 0, 0), -1)

            for i, line in enumerate(lines):
                ty = y0 + y1 + i * dy
                cv2.putText(annotated_color, line, (x1, ty), font, font_scale, bbox_color, thickness, cv2.LINE_AA)

            cv2.rectangle(annotated_color, (x1, y1), (x2, y2), bbox_color, 2)
            cv2.rectangle(annotated_depth, (x1, y1), (x2, y2), bbox_color, 2)
        localization_times.extend(det_times)
            
        
        tot_img_time = yolo_times[idx] + sum(det_times)
        print(f"Processed img [{idx+1}/48] in {tot_img_time:.3f}ms")
        
            
        make_a_movie_ts_images.append(annotated_color)
        idx+=1
        
        
        
    tot_time_full = sum(yolo_times) + sum(localization_times)
    print(f"\nProcessed all images in {tot_time_full:.1f}s")
    fps = (idx+1)/tot_time_full
    print(f"FPS: {fps:.2f}")
    
    print(f"Avg time per image: {1e3*(1/fps):.1f}ms")
    
    
    print(f"\nCreating GIF . . .", end='')
    durations = []
    for timestamp in data.keys():
        durations.append(data[timestamp]['duration'])
    
    durations = np.asarray(durations, dtype=np.float64)

    durations *= 1e-6
    frames = [Image.fromarray(cv2.cvtColor(img.astype("uint8"), cv2.COLOR_BGR2RGB)) for img in make_a_movie_ts_images]
    
    frames[0].save(
        "out.gif",
        save_all=True,
        append_images=frames[1:],
        duration=durations.tolist(),
        loop=0
    )
    print(f"\rResult GIF saved to \'out.gif\'!")
    
    try:
        cap = cv2.VideoCapture("out.gif")

        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            cv2.imshow("result gif", frame)
            if cv2.waitKey(210) & 0xFF == 27: # 210ms time, close with esc
                break

        cap.release()
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        pass
    except:
        print("Failed to open gif :-(")

if __name__ == '__main__':

    main()