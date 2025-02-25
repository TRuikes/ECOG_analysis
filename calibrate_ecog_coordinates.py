import cv2
import pandas as pd
import os
import screeninfo

# Global variables
points = []
labels = ["line_point_1", "line_point_2", "grid_top_left", "grid_top_right", "grid_bottom_right", "grid_bottom_left"]
current_index = 0
image = None
window_name = "Annotation"
directory_path = r"C:\axorus\250120-PEV_test"  # Specify the directory containing images


def click_event(event, x, y, flags, param):
    global current_index, points

    if event == cv2.EVENT_LBUTTONDOWN and current_index < len(labels):
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(image, labels[current_index], (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
        current_index += 1
        reload_image()

        if current_index == len(labels):
            save_points()

    elif event == cv2.EVENT_RBUTTONDOWN and points:
        points.pop()
        current_index -= 1
        reload_image()


def reload_image():
    global image
    image = cv2.imread(image_path)
    for i, (x, y) in enumerate(points):
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(image, labels[i], (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow(window_name, image)


def save_points():
    global image_path
    df = pd.DataFrame(points, columns=["x", "y"], index=labels[:len(points)])
    csv_path = os.path.splitext(image_path)[0] + ".csv"
    df.to_csv(csv_path)
    print(f"Points saved to {csv_path}")
    cv2.destroyAllWindows()


def process_images():
    global image_path, image, current_index, points

    if not os.path.isdir(directory_path):
        print("Invalid directory path.")
        return

    images = [f for f in os.listdir(directory_path) if "pos" in f.lower() and f.endswith((".jpg", ".png", ".jpeg"))]

    if not images:
        print("No matching images found.")
        return

    for img_file in images:
        image_path = os.path.join(directory_path, img_file)
        points = []
        current_index = 0

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            continue

        screen = screeninfo.get_monitors()[0]
        screen_width, screen_height = screen.width, screen.height

        img_height, img_width = image.shape[:2]
        window_width = screen_width // 2
        aspect_ratio = img_height / img_width
        window_height = int(window_width * aspect_ratio)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_width, window_height)
        cv2.moveWindow(window_name, (screen_width - window_width) // 2, (screen_height - window_height) // 2)

        cv2.imshow(window_name, image)
        cv2.setMouseCallback(window_name, click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    process_images()
