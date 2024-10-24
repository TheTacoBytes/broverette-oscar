import cv2
import os
import glob
import sys

def create_video_from_images(image_folder, output_video_file, fps=15, frame_size=(640, 480)):
    image_files = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))
    if len(image_files) == 0:
        print(f"No images found in folder {image_folder}")
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, frame_size)

    for image_file in image_files:
        img = cv2.imread(image_file)

        if img.shape[1] != frame_size[0] or img.shape[0] != frame_size[1]:
            img = cv2.resize(img, frame_size)

        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (10, 30)  # Position of the text on the frame
        font_scale = 1
        color = (255, 0, 0)  
        thickness = 2
        img = cv2.putText(img, os.path.basename(image_file), position, font, font_scale, color, thickness, cv2.LINE_AA)

       
        video_writer.write(img)

    video_writer.release()
    print(f"Video file saved as {output_video_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python3 create_video.py <path_to_image_folder>')
        sys.exit(1)

    image_folder = sys.argv[1]
    
    current_directory = os.getcwd()
    print(f"Running from directory: {current_directory}")

    folder_name = os.path.basename(os.path.normpath(image_folder))
    
    video_dir = os.path.join(current_directory, 'video')

    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
        print(f"Created directory: {video_dir}")
    
    output_video_file = os.path.join(video_dir, f'{folder_name}.mp4')

    fps = 15  # Frames per second for the video
    frame_size = (640, 480)  # Frame size based on your images (width, height)

    create_video_from_images(image_folder, output_video_file, fps, frame_size)
    
    sys.exit(0)
