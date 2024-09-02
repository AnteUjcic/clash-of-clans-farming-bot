
import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN
import random
from scipy.spatial import distance
import time
import win32gui, win32ui, win32con, win32api

# Screen capture function
def capture_screen(region=None):
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, right, bot = region
        width = right - left
        height = bot - top
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
    
    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    bmpinfo = bmp.GetInfo()
    bmpstr = bmp.GetBitmapBits(True)

    img = np.frombuffer(bmpstr, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv.cvtColor(img, cv.COLOR_BGRA2BGR)

# Non-maximum suppression function
def non_max_suppression(boxes, overlap_thresh):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - y1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    return boxes[pick].astype("int")

# Function to detect and draw red outlines
def detect_and_draw_red_outlines(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    lower_red1 = np.array([10, 103, 100])
    upper_red1 = np.array([14, 150, 140])
    lower_red2 = np.array([10, 1, 1])
    upper_red2 = np.array([68, 20, 12])

    mask1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2

    contours, _ = cv.findContours(red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image, contours, -1, (0, 0, 255), 2)

    return contours

# Function to calculate the distance from point to rectangle edges
def distance_from_rect(pt, rect):
    cx, cy = pt
    x1, y1, x2, y2 = rect
    dx = max(x1 - cx, 0, cx - x2)
    dy = max(y1 - cy, 0, cy - y2)
    return np.sqrt(dx*dx + dy*dy)

def min_distance_to_contours(point, contours):
    min_distance = float('inf')
    for contour in contours:
        for contour_point in contour:
            distance_to_point = distance.euclidean(point, contour_point[0])
            if distance_to_point < min_distance:
                min_distance = distance_to_point
    return min_distance

def min_distance_to_rectangles_in_cluster(point, cluster_rectangles):
    min_distance = float('inf')
    for rect in cluster_rectangles:
        rect_distance = distance_from_rect(point, rect)
        if rect_distance < min_distance:
            min_distance = rect_distance
    return min_distance

# Initialize video writer
output_file = 'output_video.avi'
frame_width = 1920
frame_height = 1080
#fourcc = cv.VideoWriter_fourcc(*'XVID')  # Specify the codec
#out = cv.VideoWriter(output_file, fourcc, 20.0, (frame_width, frame_height))

# Main processing loop
while True:
    # Capture the screen
    img_rgb = capture_screen(region=(0, 0, 1920, 1080))
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

    # Template paths and matching
    template_paths = {
    'elixir': [
        r'cocimg\elixir\elixir1.png', r'cocimg\elixir\elixir2.png', r'cocimg\elixir\elixir3.png', 
        r'cocimg\elixir\elixir4.png', r'cocimg\elixir\elixir5.png', r'cocimg\elixir\elixir6.png', 
        r'cocimg\elixir\elixir7.png', r'cocimg\elixir\elixir8.png', r'cocimg\elixir\elixir9.png', 
        r'cocimg\elixir\elixir10.png', r'cocimg\elixir\elixir11.png', r'cocimg\elixir\elixir12.png', 
        r'cocimg\elixir\elixir13.png'
    ],
    'gold': [
        r'cocimg\gold\gold1.png', r'cocimg\gold\gold2.png', r'cocimg\gold\gold3.png', r'cocimg\gold\gold4.png', 
        r'cocimg\gold\gold5.png', r'cocimg\gold\gold6.png', r'cocimg\gold\gold7.png', r'cocimg\gold\gold8.png', 
        r'cocimg\gold\gold9.png', r'cocimg\gold\gold10.png', r'cocimg\gold\gold11.png', r'cocimg\gold\gold12.png', 
        r'cocimg\gold\gold13.png', r'cocimg\gold\gold14.png', r'cocimg\gold\gold15.png', r'cocimg\gold\gold16.png', 
        r'cocimg\gold\gold17.png'
    ],
    'darkelixir': [
        r'cocimg\darkelixir\darkelixir1.png', r'cocimg\darkelixir\darkelixir2.png', r'cocimg\darkelixir\darkelixir3.png', 
        r'cocimg\darkelixir\darkelixir4.png', r'cocimg\darkelixir\darkelixir5.png', r'cocimg\darkelixir\darkelixir6.png', 
        r'cocimg\darkelixir\darkelixir7.png', r'cocimg\darkelixir\darkelixir8.png'
    ]
}

    threshold = 0.67
    all_boxes = []

    for template_type, paths in template_paths.items():
        for template_path in paths:
            template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
            if template is None:
                continue

            w, h = template.shape[::-1]
            res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)

            boxes = []
            for pt in zip(*loc[::-1]):
                boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h])

            boxes = np.array(boxes)
            boxes = non_max_suppression(boxes, 0.3)

            for (x1, y1, x2, y2) in boxes:
                all_boxes.append([x1, y1, x2, y2, (x1 + x2) / 2, (y1 + y2) / 2])

    # Detect and draw red outlines
    red_contours = detect_and_draw_red_outlines(img_rgb)

    # Cluster the center points of the rectangles
    if len(all_boxes) > 0:
        all_boxes = np.array(all_boxes)
        clustering = DBSCAN(eps=90, min_samples=1).fit(all_boxes[:, 4:])
        labels = clustering.labels_

        unique_labels = set(labels)
        centroids = []

        colors = {label: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for label in unique_labels}

        for label in unique_labels:
            if label == -1:
                continue
            label_mask = (labels == label)
            cluster_points = all_boxes[label_mask]
            cluster_rectangles = cluster_points[:, :4]
            centroid = cluster_points[:, 4:].mean(axis=0)

            valid = False
            while not valid:
                valid = True
                for point in cluster_points[:, 4:]:
                    if distance_from_rect(centroid, [point[0]-15, point[1]-15, point[0]+15, point[1]+15]) < 30:
                        centroid += np.random.uniform(-30, 30, size=2)
                        valid = False
                        break
            centroids.append(centroid)

            for (x1, y1, x2, y2, cx, cy) in cluster_points:
                color = colors[label]
                cv.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        for centroid in centroids:
            cx, cy = int(centroid[0]), int(centroid[1])
            cv.circle(img_rgb, (cx, cy), 5, (0, 255, 0), -1)
            cv.circle(img_rgb, (cx, cy), 150, (0, 255, 0), 2)

            # Simulate mouse clicks around the circumference
            radius = 150
            step = 40
            num_points = int(2 * np.pi * radius / step)
            circumference_points = []

            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                x = int(cx + radius * np.cos(angle))
                y = int(cy + radius * np.sin(angle))
                circumference_points.append((x, y))

            for point in circumference_points:
                min_dist = min_distance_to_contours(point, red_contours)
                min_dist_rect = min_distance_to_rectangles_in_cluster(point, cluster_rectangles)
                if min_dist <= 30 and min_dist_rect <= 50:
                    print(f"Simulating click at {point} (distance to nearest contour: {min_dist:.2f}, rect: {min_dist_rect:.2f})")
                    cv.circle(img_rgb, point, 5, (255, 0, 0), -1)

    # Write the processed frame to the video file
    #out.write(img_rgb)

    # Display the image
    cv.imshow('Detected Objects', img_rgb)

    # Break loop on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video writer and close OpenCV windows
#out.release()
cv.destroyAllWindows()
