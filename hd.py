import sqlite3
import subprocess
import os
import time
import requests
import uuid
import shutil
import platform
import random
import pathlib
import cv2
import numpy as np
import traceback
from playwright.sync_api import sync_playwright

# --- CONFIGURATION ---
DB_FILE = "media.db"
INPUT_FOLDER = "inputs"
TEMP_FOLDER = "temp"
TEMPLATES_FOLDER = "templates"
ANIMATIONS_FILE = "animations.txt"

FFMPEG_PATH = "ffmpeg"  # Use system FFmpeg from PATH

STANDARDIZED_RESOLUTION = "1080x1920"
STANDARDIZED_FRAMERATE = 24
HTML_FRAMERATE = 24  # Frame rate for HTML animations with embedded videos
# Note: This should match STANDARDIZED_FRAMERATE for consistent timing

PEXELS_API_KEYS = ["ZpI0W9YqXBGOmJ2GVOQupu9geeo85c81TZF2WM9nJZ0gF4w6Ox5dERe1", "eQtvuGdNTzMImaM7hr1C0WGalAZYarmfgcFr30s9VCrnE4MLGFm6Lo2S"]
PIXABAY_API_KEYS = ["52161921-e0d0dd75191a119f81f3e9a6e", "13344532-52325730428538ad607a52ec7"]

current_pexels_key_index = 0
current_pixabay_key_index = 0

# ==============================================================================
# === CORRECTED GPU ENCODER DETECTION FUNCTION =================================
# ==============================================================================
def get_best_encoder():
    """
    Checks for available hardware encoders using a safe resolution and returns the best one.
    """
    print("-> Detecting available GPU encoders...")
    # Using a larger, safer resolution for the test (e.g., 640x360) to avoid errors
    # on NVENC hardware which has a minimum resolution requirement.
    test_resolution = "640x360"
    
    encoders = {
        'h264_nvenc': {'name': 'NVIDIA NVENC', 'cmd': [FFMPEG_PATH, '-y', '-f', 'lavfi', '-i', f'nullsrc=s={test_resolution}:d=1', '-c:v', 'h264_nvenc', '-preset', 'p1', '-f', 'null', '-']},
        'h264_qsv': {'name': 'Intel QSV', 'cmd': [FFMPEG_PATH, '-y', '-f', 'lavfi', '-i', f'nullsrc=s={test_resolution}:d=1', '-c:v', 'h264_qsv', '-preset', 'veryfast', '-f', 'null', '-']},
        'h264_amf': {'name': 'AMD AMF', 'cmd': [FFMPEG_PATH, '-y', '-f', 'lavfi', '-i', f'nullsrc=s={test_resolution}:d=1', '-c:v', 'h264_amf', '-quality', 'speed', '-f', 'null', '-']}
    }

    for codec, details in encoders.items():
        try:
            subprocess.run(details['cmd'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            print(f"   SUCCESS: Found {details['name']} ({codec}). Will use this for encoding.")
            return codec
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"   - {details['name']} not available.")

    print("   -> No compatible GPU encoder found. Falling back to CPU (libx264).")
    return 'libx264'

# --- HELPER FUNCTIONS ---
def time_str_to_seconds(time_str):
    if not time_str: return 0
    time_str = time_str.replace(',', '.')
    parts = time_str.split(':')
    if len(parts) == 3: # HH:MM:SS.ms
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2: # MM:SS.ms
        return float(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 1: # SS.ms
        return float(parts[0])
    return 0

def get_next_api_key(keys, index_ref):
    key = keys[index_ref['index']]
    index_ref['index'] = (index_ref['index'] + 1) % len(keys)
    return key

def is_greenscreen_video(video_path, threshold=0.8):
    """
    Checks if a video is likely a greenscreen video by checking the corners of the first frame.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return False

        h, w, _ = frame.shape
        corners = [
            frame[0:10, 0:10],  # Top-left
            frame[0:10, w-10:w],  # Top-right
            frame[h-10:h, 0:10],  # Bottom-left
            frame[h-10:h, w-10:w]  # Bottom-right
        ]

        green_count = 0
        for corner in corners:
            # Check if the average color is green
            avg_color = np.mean(corner, axis=(0, 1))
            # In BGR, green is the second channel
            if avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:
                green_count += 1
        
        return (green_count / len(corners)) >= threshold
    except Exception as e:
        print(f"Error checking for greenscreen: {e}")
        return False

def is_text_present(video_path, frame_skip=30, confidence_threshold=5):
    """
    Checks if a video likely contains prominent graphical text by analyzing a few frames.
    Aims to detect overlays, not small scene text.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        text_detections = 0
        
        # Limit frames to check for performance, e.g., max 10 frames
        frames_to_check = [i for i in range(0, frame_count, frame_skip)][:10]

        for frame_index in frames_to_check:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                continue

            # Pre-processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Apply morphological gradient to find areas with high contrast (like text edges)
            kernel = np.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

            # Binarize
            _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # Connected components to find letter-like regions
            # Thicker kernel for connecting parts of letters
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)) # horizontal kernel
            connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours based on text characteristics
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Heuristics for text blocks (could be a single word or a line)
                if (w > 20 and h > 10 and # reasonable size
                    0.5 < (w / float(h)) < 20 and # aspect ratio of a word/line
                    cv2.contourArea(contour) > 200):
                    text_detections += 1
            
            # If we find enough potential text in one frame, we can stop early
            if text_detections >= confidence_threshold:
                cap.release()
                print(f"  - Detected {text_detections} potential text regions in a frame. Rejecting.")
                return True

        cap.release()
        return False

    except Exception as e:
        print(f"Error during text detection: {e}")
        return False

def download_file(url, destination):
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(destination, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        
        if destination.endswith(".mp4") and is_greenscreen_video(destination):
            print(f"  - Greenscreen video detected. Discarding {destination}")
            os.remove(destination)
            return False

        time.sleep(1)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False

def run_ffmpeg_command(command, is_blocking=True):
    try:
        if is_blocking:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        else:
            return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"--- FFmpeg Error ---\nCommand failed: {' '.join(command)}\nFFmpeg stderr: {e.stderr.decode()}\n--------------------")
        raise
    except Exception as e:
        print(f"Failed to start command: {' '.join(command)} - Error: {e}")
        return None

# --- SEARCH FUNCTIONS ---
def search_in_database(keywords, media_type):
    try:
        conn = sqlite3.connect(DB_FILE); cursor = conn.cursor()
        params = [f"%{term}%" for term in keywords]
        if media_type == 'video':
            query = "SELECT s.id, m.storage_location, s.start, s.end FROM segments s JOIN media m ON s.media_id = m.id WHERE " + " AND ".join(["(s.label || ' ' || s.tags) LIKE ?"] * len(keywords)) + " LIMIT 1"
        elif media_type == 'image':
            query = "SELECT id, storage_location FROM media WHERE media_type = 'image' AND " + " AND ".join(["name LIKE ?"] * len(keywords)) + " LIMIT 1"
        else: conn.close(); return None
        cursor.execute(query, params); result = cursor.fetchone()
        conn.close()
        if result:
            item_id = result[0]
            if media_type == 'video': return {"id": f"db_{item_id}", "path": result[1], "start": result[2], "end": result[3]}
            else: return {"id": f"db_{item_id}", "path": result[1]}
        return None
    except sqlite3.Error as e: print(f"Database error: {e}"); return None

def search_pexels(keywords, media_type):
    global current_pexels_key_index
    index_ref = {'index': current_pexels_key_index}
    api_key = get_next_api_key(PEXELS_API_KEYS, index_ref)
    current_pexels_key_index = index_ref['index']
    search_type, url = ('videos', f"https://api.pexels.com/videos/search") if media_type == 'video' else ('photos', "https://api.pexels.com/search")
    params = {"query": " ".join(keywords), "per_page": 15, "orientation": "landscape"}
    try:
        response = requests.get(url, headers={"Authorization": api_key}, params=params); response.raise_for_status(); data = response.json()
        results = data.get(search_type)
        if results:
            item = random.choice(results)
            item_id = f"pexels_{item['id']}"
            if media_type == 'video':
                for vf in item.get('video_files', []):
                    # Prioritize HD resolution but accept others
                    if vf.get('width') >= 1280: 
                        return {"id": item_id, "url": vf['link']}
                return {"id": item_id, "url": item['video_files'][0]['link']}
            else: return {"id": item_id, "url": item['src'].get('large2x')}
        return None
    except requests.exceptions.RequestException as e: print(f"Pexels API error: {e}"); return None

def search_pixabay(keywords, media_type):
    global current_pixabay_key_index
    index_ref = {'index': current_pixabay_key_index}
    api_key = get_next_api_key(PIXABAY_API_KEYS, index_ref)
    current_pixabay_key_index = index_ref['index']
    search_type = 'videos' if media_type == 'video' else 'photos'
    url = f"https://pixabay.com/api/{'videos/' if media_type == 'video' else ''}"
    params = {"key": api_key, "q": " ".join(keywords), "per_page": 20, "orientation": "horizontal"}
    try:
        response = requests.get(url, params=params); response.raise_for_status(); data = response.json()
        results = data.get('hits')
        if results:
            item = random.choice(results)
            item_id = f"pixabay_{item['id']}"
            if media_type == 'video': return {"id": item_id, "url": item['videos']['large']['url']}
            else: return {"id": item_id, "url": item['largeImageURL']}
        return None
    except requests.exceptions.RequestException as e: print(f"Pixabay API error: {e}"); return None

def find_image_file_for_depthflow(image_name: str, images_folder: str = "images") -> str:
    """Find image file with given name in the images folder for depthflow."""
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    for ext in extensions:
        image_path = os.path.join(images_folder, f"{image_name}{ext}")
        if os.path.exists(image_path):
            return image_path
    return None

def load_depthflow_styles(style_file: str = "images/img video style.txt") -> list:
    """Load depthflow command templates from style file."""
    styles = []
    if os.path.exists(style_file):
        with open(style_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and line.startswith('depthflow'):
                    styles.append(line)
    return styles

def generate_depthflow_video(task):
    """Generate video using depthflow for imgvideo tasks."""
    print(f"\n-> Generating depthflow video for Task {task['id']}: {task['search_terms']}")
    
    # Load depthflow styles
    styles = load_depthflow_styles()
    if not styles:
        print("  ERROR: No depthflow styles found in 'images/img video style.txt'")
        return None
    
    # Get the image name from search terms
    image_name = task['search_terms'][0] if task['search_terms'] else ''
    
    # Find the image file
    image_path = find_image_file_for_depthflow(image_name)
    if not image_path:
        print(f"  ERROR: Image '{image_name}' not found in images folder")
        return None
    
    print(f"  - Found image: {image_path}")
    
    # Randomly select a style template
    style_template = random.choice(styles)
    print(f"  - Selected random style: {styles.index(style_template) + 1} of {len(styles)}")
    
    # Generate output filename
    output_name = f"depthflow_{task['id']}_{image_name.replace(' ', '_')}.mp4"
    output_path = os.path.join(TEMP_FOLDER, output_name)
    
    # Replace thumbnail.jpg with actual image path (quote it for spaces)
    quoted_image_path = f'"{image_path}"'
    command = style_template.replace('thumbnail.jpg', quoted_image_path)
    
    # Replace time parameter with calculated duration
    import re
    time_pattern = r'--time\s+\d+'
    command = re.sub(time_pattern, f'--time {task["duration"]}', command)
    
    # Replace output parameter
    escaped_output_path = output_path.replace('\\', '/')
    command = re.sub(r'--output\s+\S+', f'--output "{escaped_output_path}"', command)
    
    print(f"  - Running depthflow command...")
    print(f"    {command}")
    
    try:
        # Run the depthflow command
        import subprocess
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(output_path):
            print(f"  SUCCESS: Depthflow video generated at {output_path}")
            return {
                "id": f"depthflow_{task['id']}",
                "source_path": output_path,
                "type": "depthflow_video",
                "duration": task['duration']
            }
        else:
            print(f"  ERROR: Depthflow command failed")
            print(f"  stdout: {result.stdout}")
            print(f"  stderr: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"  ERROR: Failed to run depthflow command: {e}")
        return None

def get_video_duration(video_path):
    """Get the duration of a video file in seconds."""
    try:
        cmd = [FFMPEG_PATH, '-i', video_path, '-f', 'null', '-']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse duration from ffmpeg stderr output
        import re
        duration_match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})', result.stderr)
        if duration_match:
            hours, minutes, seconds = duration_match.groups()
            total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            return total_seconds
        return 0
    except Exception as e:
        print(f"    - Could not get video duration: {e}")
        return 0

def find_multiple_clips_for_duration(keywords, media_type, required_duration, task_id):
    """Find multiple clips to fill the required duration."""
    clips = []
    total_duration = 0
    attempt = 0
    max_attempts = 5
    
    print(f"    - Need {required_duration}s of footage, searching for multiple clips...")
    
    while total_duration < required_duration and attempt < max_attempts:
        attempt += 1
        print(f"    - Attempt {attempt}: Looking for additional clips...")
        
        # Try Pixabay first
        pixabay_result = search_pixabay(keywords, media_type)
        if pixabay_result:
            file_ext = ".mp4"
            dl_path = os.path.join(TEMP_FOLDER, f"download_{task_id}_part{attempt}{file_ext}")
            if download_file(pixabay_result['url'], dl_path):
                duration = get_video_duration(dl_path)
                if duration > 0:
                    clips.append({"path": dl_path, "duration": duration})
                    total_duration += duration
                    print(f"    - Downloaded clip {attempt}: {duration:.1f}s (total: {total_duration:.1f}s)")
                    if total_duration >= required_duration:
                        break
        
        # Try Pexels if still need more
        if total_duration < required_duration:
            pexels_result = search_pexels(keywords, media_type)
            if pexels_result:
                file_ext = ".mp4"
                dl_path = os.path.join(TEMP_FOLDER, f"download_{task_id}_pexels{attempt}{file_ext}")
                if download_file(pexels_result['url'], dl_path):
                    duration = get_video_duration(dl_path)
                    if duration > 0:
                        clips.append({"path": dl_path, "duration": duration})
                        total_duration += duration
                        print(f"    - Downloaded clip {attempt}b: {duration:.1f}s (total: {total_duration:.1f}s)")
    
    if clips and total_duration >= required_duration:
        # Concatenate clips into one file
        combined_path = os.path.join(TEMP_FOLDER, f"combined_{task_id}.mp4")
        concat_list_path = os.path.join(TEMP_FOLDER, f"concat_{task_id}.txt")
        
        with open(concat_list_path, 'w') as f:
            for clip in clips:
                f.write(f"file '{os.path.abspath(clip['path'])}'\n")
        
        try:
            cmd = [FFMPEG_PATH, '-y', '-f', 'concat', '-safe', '0', '-i', concat_list_path, '-c', 'copy', combined_path]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            
            print(f"    SUCCESS: Combined {len(clips)} clips into {total_duration:.1f}s video")
            return {"id": f"combined_{task_id}", "source_path": combined_path, "type": "downloaded_video", "duration": total_duration}
        except Exception as e:
            print(f"    - Failed to combine clips: {e}")
    
    return None

def find_footage(task):
    # Handle imgvideo tasks with depthflow
    if task['media_type'] == 'imgvideo':
        return generate_depthflow_video(task)
    
    print(f"\n-> Searching for Task {task['id']} ({task['media_type']}): {task['search_terms']}")
    required_duration = task.get('duration', 5)
    
    for term_group in task['search_terms']:
        keywords = term_group.split(); print(f"  - Trying term: '{term_group}'...")

        # New: First, check for local image if media_type is image
        if task['media_type'] == 'image':
            print(f"    - Searching for local image '{term_group}' in 'images/' folder...")
            for ext in ['.jpg', '.jpeg', '.png']:
                local_image_path = os.path.join('images', f"{term_group}{ext}")
                if os.path.exists(local_image_path):
                    print(f"    SUCCESS: Found local image at '{local_image_path}'.")
                    return {
                        "id": f"local_{term_group.replace(' ', '_')}",
                        "source_path": local_image_path,
                        "type": "local_image"
                    }
            print(f"    - Local image not found. Checking databases...")

        local_item = search_in_database(keywords, task['media_type'])
        if local_item:
            print(f"    SUCCESS: Found '{term_group}' in local DB.")
            if task['media_type'] == 'video':
                duration = local_item.get('end', 0) - local_item.get('start', 0)
                return {"id": local_item['id'], "source_path": local_item['path'], "start": local_item.get('start'), "end": local_item.get('end'), "type": f"local_{task['media_type']}", "duration": duration}
            else:
                return {"id": local_item['id'], "source_path": local_item['path'], "type": f"local_{task['media_type']}"}
        
        # For video tasks, try to find clips that meet duration requirements
        if task['media_type'] == 'video':
            print(f"    - Searching for {required_duration}s of video footage...")
            clips_collected = []
            total_duration = 0
            
            # Try to collect clips until we have enough duration
            sources = ['pixabay', 'pexels']
            for source in sources:
                if total_duration >= required_duration:
                    break
                    
                print(f"    - Trying {source}...")
                if source == 'pixabay':
                    result = search_pixabay(keywords, task['media_type'])
                else:
                    result = search_pexels(keywords, task['media_type'])
                
                if result:
                    file_ext = ".mp4"
                    dl_path = os.path.join(TEMP_FOLDER, f"download_{task['id']}_{source}{file_ext}")
                    if download_file(result['url'], dl_path):
                        duration = get_video_duration(dl_path)
                        if duration > 0:
                            clips_collected.append({"path": dl_path, "duration": duration, "id": result['id']})
                            total_duration += duration
                            print(f"    - Downloaded {duration:.1f}s clip (total: {total_duration:.1f}s)")
                        else:
                            print(f"    - Could not determine duration, removing file")
                            if os.path.exists(dl_path):
                                os.remove(dl_path)
            
            # Check if we have enough footage
            if clips_collected and total_duration >= required_duration:
                if len(clips_collected) == 1:
                    # Single clip is enough
                    clip = clips_collected[0]
                    print(f"    SUCCESS: Single clip with {clip['duration']:.1f}s (need {required_duration}s)")
                    return {"id": clip['id'], "source_path": clip['path'], "type": "downloaded_video", "duration": clip['duration']}
                else:
                    # Need to combine multiple clips
                    print(f"    - Combining {len(clips_collected)} clips for {total_duration:.1f}s total")
                    combined_path = os.path.join(TEMP_FOLDER, f"combined_{task['id']}.mp4")
                    concat_list_path = os.path.join(TEMP_FOLDER, f"concat_{task['id']}.txt")
                    
                    with open(concat_list_path, 'w') as f:
                        for clip in clips_collected:
                            f.write(f"file '{os.path.abspath(clip['path'])}'\n")
                    
                    try:
                        cmd = [FFMPEG_PATH, '-y', '-f', 'concat', '-safe', '0', '-i', concat_list_path, '-c', 'copy', combined_path]
                        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                        
                        print(f"    SUCCESS: Combined clips into {total_duration:.1f}s video")
                        return {"id": f"combined_{task['id']}", "source_path": combined_path, "type": "downloaded_video", "duration": total_duration}
                    except Exception as e:
                        print(f"    - Failed to combine clips: {e}")
            
            # If we still don't have enough, try to get more clips
            if clips_collected and total_duration < required_duration:
                print(f"    - Need more footage, trying additional searches...")
                additional_clips = find_multiple_clips_for_duration(keywords, task['media_type'], required_duration - total_duration, f"{task['id']}_extra")
                if additional_clips:
                    return additional_clips
        
        else:
            # For non-video tasks, use original logic
            print(f"    - Not in DB. Searching Pixabay...")
            pixabay_result = search_pixabay(keywords, task['media_type'])
            if pixabay_result:
                print(f"    SUCCESS: Found '{term_group}' on Pixabay. Downloading...")
                file_ext = ".jpg" if task['media_type'] == 'image' else ".mp4"
                dl_path = os.path.join(TEMP_FOLDER, f"download_{task['id']}{file_ext}")
                if download_file(pixabay_result['url'], dl_path): return {"id": pixabay_result['id'], "source_path": dl_path, "type": f"downloaded_{task['media_type']}"}

            print(f"    - Not on Pixabay. Searching Pexels...")
            pexels_result = search_pexels(keywords, task['media_type'])
            if pexels_result:
                print(f"    SUCCESS: Found '{term_group}' on Pexels. Downloading...")
                file_ext = ".jpg" if task['media_type'] == 'image' else ".mp4"
                dl_path = os.path.join(TEMP_FOLDER, f"download_{task['id']}{file_ext}")
                if download_file(pexels_result['url'], dl_path): return {"id": pexels_result['id'], "source_path": dl_path, "type": f"downloaded_{task['media_type']}"}
        
    print(f"  FAILURE: No footage found for any search terms for task {task['id']}.")
    return None

# --- TEMPLATE SYSTEM FUNCTIONS ---
def parse_animations_file(filepath):
    animations_db = {}
    if not os.path.exists(filepath):
        print(f"Warning: Animations file not found at '{filepath}'")
        return animations_db
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 3: continue
            
            anim_id, template_folder, data_parts = parts[0], parts[1], parts[2:]
            data = {}
            for item in data_parts:
                if '=' in item:
                    key, value = item.split('=', 1)
                    data[key.strip()] = value.strip()
            
            animations_db[anim_id] = {"template": template_folder, "data": data}
            
    print(f"-> Loaded {len(animations_db)} definitions from {os.path.basename(filepath)}")
    return animations_db

def parse_input_file(filepath):
    tasks = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#') or '|' not in line:
                continue
            try:
                parts = [p.strip() for p in line.split('|')]
                task_id = parts[0]
                time_range = parts[1].split('-')
                start_sec = time_str_to_seconds(time_range[0])
                end_sec = time_str_to_seconds(time_range[1])
                duration = end_sec - start_sec
                media_type = parts[2].lower()
    
                # Save absolute start and end times
                task_data = {
                    "id": task_id,
                    "duration": duration,
                    "media_type": media_type,
                    "start_time": start_sec,
                    "end_time": end_sec
                }
    
                if media_type == 'html':
                    # For html tasks, the 4th column is the animation ID.
                    task_data["animation_id"] = parts[3]
                elif media_type == 'image' and len(parts) >= 5:
                    # This is our new image animation format: image | name | animation_id
                    task_data["media_type"] = 'html' # Treat it as an html task for rendering
                    task_data["is_image_animation"] = True # Add a flag
                    task_data["search_terms"] = [parts[3]] # The image name, e.g., 'as'
                    task_data["animation_id"] = parts[4] # The animation style, e.g., 'zoom-in-photo'

                elif media_type in ['video', 'image', 'imgvideo']:
                    extra = parts[3]
                    greenscreen = False
                    nogreenscreen = False
                    search_terms = []
                    
                    # Check if the extra field contains a dash indicating greenscreen/nogreenscreen flag.
                    if ' - ' in extra:
                        search_field, flag_field = extra.split(' - ', 1)
                        flag = flag_field.strip().lower()
                        
                        if flag == "greenscreen":
                            greenscreen = True
                        elif flag == "nogreenscreen":
                            nogreenscreen = True
                        
                        search_terms = [term.strip() for term in search_field.split(',')]
                    else:
                        search_terms = [term.strip() for term in extra.split(',')]
                    
                    # For nogreenscreen, check if template is specified in the next column
                    if nogreenscreen and len(parts) >= 5:
                        task_data["template_style"] = parts[4].strip()
                        # Also parse any additional key=value parameters for nogreenscreen
                        if len(parts) > 5:
                            params = {}
                            for field in parts[5:]:
                                if '=' in field:
                                    key, value = field.split('=', 1)
                                    params[key.strip()] = value.strip()
                            task_data["params"] = params

                    # Handle the simple image search case: | image | crying boy |
                    if not greenscreen and not nogreenscreen and len(parts) == 4:
                        pass # search_terms is already correctly set to [extra]
    
                    task_data["search_terms"] = search_terms
                    task_data["nogreenscreen"] = nogreenscreen
    
                    # If greenscreen is used or additional overlay properties exist,
                    # process extra columns as overlay details.
                    if greenscreen:
                        overlay_details = {}
                        # Next column (if exists) represents the overlay style or ID.
                        if len(parts) >= 5:
                            overlay_details["animation"] = parts[4]
                        # Any further columns are parsed as key=value pairs.
                        if len(parts) > 5:
                            for field in parts[5:]:
                                if '=' in field:
                                    key, value = field.split('=', 1)
                                    overlay_details[key.strip()] = value.strip()
                        # Mark that a greenscreen effect should be applied.
                        overlay_details["greenscreen"] = "true"
                        task_data["overlay"] = overlay_details
                else:
                    continue
                tasks.append(task_data)
            except Exception as e: 
                print(f"Warning: Could not parse line {idx+1} in {os.path.basename(filepath)}: '{line}' - Error: {e}")
    return tasks

def render_overlay(task, animations_db, encoder, duration_override=None, group_id=None):
    """
    Renders an overlay animation using PNG screenshots with transparency at 10 FPS.
    """
    overlay_details = task.get("overlay", {})
    anim_id = overlay_details.get("animation", overlay_details.get("id"))
    if not anim_id:
        print(f"  ERROR: No animation ID specified for overlay in task {task['id']}.")
        return None

    anim_data = animations_db.get(anim_id)
    if not anim_data:
        print(f"  ERROR: Animation ID '{anim_id}' not found in animations definitions.")
        return None

    template_folder = anim_data['template']
    template_path = os.path.join(TEMPLATES_FOLDER, template_folder)
    if not os.path.isdir(template_path):
        print(f"  ERROR: Template folder '{template_folder}' not found for overlay.")
        return None

    render_id = group_id if group_id else task['id']
    record_duration = duration_override if duration_override is not None else task.get('duration', 0)
    
    # Create frames directory
    frames_dir = os.path.join(TEMP_FOLDER, f"overlay_frames_{render_id}")
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)

    try:
        # Prepare HTML content
        base_html_path = os.path.join(template_path, 'index.html')
        with open(base_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        for key, value in {**anim_data['data'], **overlay_details}.items():
            html_content = html_content.replace(f"{{{{{key}}}}}", value)
        html_content = html_content.replace("background:", "background:#00B140;")
        
        render_html_path = os.path.join(TEMP_FOLDER, f"overlay_render_{render_id}.html")
        with open(render_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Copy assets
        for filename in os.listdir(template_path):
            source_file = os.path.join(template_path, filename)
            if os.path.isfile(source_file):
                shutil.copy(source_file, TEMP_FOLDER)

    except Exception as e:
        print(f"  ERROR: Failed to prepare overlay HTML template: {e}")
        return None

    # Capture frames at 10 FPS
    fps = 10
    total_frames = int(record_duration * fps)
    frame_interval = 1000 / fps  # milliseconds between frames

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, channel="chrome")
            page = browser.new_page(viewport={'width': 1920, 'height': 1080})
            page.goto(pathlib.Path(os.path.abspath(render_html_path)).as_uri())
            page.wait_for_load_state('domcontentloaded', timeout=10000)

            print(f"  - Capturing {total_frames} transparent frames at {fps} FPS...")
            for frame_num in range(total_frames):
                screenshot_path = os.path.join(frames_dir, f"frame_{frame_num:05d}.png")
                # Take screenshot with transparency enabled
                page.screenshot(
                    path=screenshot_path,
                    type='png',  # PNG for transparency support
                    omit_background=True  # Enable transparency
                )
                page.wait_for_timeout(frame_interval)
                if frame_num % 10 == 0:
                    print(f"    Progress: {frame_num}/{total_frames} frames")

            browser.close()

    except Exception as e:
        print(f"  ERROR: Failed to capture frames: {e}")
        return None

    # Combine frames into video with alpha channel support
    output_overlay_path = os.path.join(TEMP_FOLDER, f"overlay_capture_{render_id}.mov")
    try:
        cmd = [
            FFMPEG_PATH, '-y',
            '-framerate', str(fps),
            '-i', os.path.join(frames_dir, 'frame_%05d.png'),
            '-c:v', 'prores_ks',
            '-profile:v', '4444',  # ProRes 4444 for alpha support
            '-vendor', 'ap10',
            '-bits_per_mb', '8000',
            '-pix_fmt', 'yuva444p10le',  # 10-bit 4:4:4:4 pixel format with alpha
            output_overlay_path
        ]
        run_ffmpeg_command(cmd)
        
        # Clean up frames directory
        shutil.rmtree(frames_dir)
        
    except Exception as e:
        print(f"  ERROR: Failed to combine frames into video: {e}")
        return None

    print(f"  SUCCESS: Transparent overlay animation recorded to {output_overlay_path}")
    return output_overlay_path

def apply_overlay(main_clip, overlay_clip, output_path, overlay_offset=0):
    """
    Composites the transparent overlay on top of the main clip.
    """
    input_options = []
    if overlay_offset > 0:
        input_options.extend(['-ss', str(overlay_offset)])

    cmd = [
        FFMPEG_PATH, '-y',
        '-i', main_clip,
        *input_options,
        '-i', overlay_clip,
        '-filter_complex',
        "[0:v][1:v]overlay=0:0",  # Simple overlay without chroma key
        '-c:v', 'mpeg2video', '-q:v', '2',
        '-c:a', 'copy',
        '-shortest',
        output_path
    ]
    run_ffmpeg_command(cmd)
    return output_path

def render_html_animation(task, animations_db, video_source=None):
    """Uses Playwright with the system's installed Chrome to fix codec issues."""
    print(f"\n-> Rendering HTML animation for Task {task['id']}' with Playwright")
    
    anim_id = task['animation_id']
    anim_data = animations_db.get(anim_id)
    if not anim_data:
        print(f"  ERROR: Animation ID '{anim_id}' not found. Cannot render.")
        return None
        
    template_folder = anim_data['template']
    template_path = os.path.join(TEMPLATES_FOLDER, template_folder)
    if not os.path.isdir(template_path):
        print(f"  ERROR: Template folder '{template_folder}' not found.")
        return None

    try:
        base_html_path = os.path.join(template_path, 'index.html')
        with open(base_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Check if it is an image animation and replace the image placeholder
        if task.get("is_image_animation"):
            # Construct the expected image path
            image_name = task['search_terms'][0] # Assuming first search term is the image name (without extension)
            
            # Define potential image paths with different extensions
            potential_image_paths = [
                os.path.join('images', f"{image_name}.jpg"),
                os.path.join('images', f"{image_name}.jpeg"),
                os.path.join('images', f"{image_name}.png")
            ]

            actual_image_path = None
            for p in potential_image_paths:
                if os.path.exists(p):
                    actual_image_path = p
                    break

            if actual_image_path:
                # Create a relative path that HTML can understand, using forward slashes
                relative_path = f"../images/{os.path.basename(actual_image_path)}" 
                # Replace the image placeholder
                html_content = html_content.replace("{{imageholder}}", relative_path)
                print(f"  - Replaced {{imageholder}} with: {relative_path}")
            else:
                print(f"  WARNING: Image '{image_name}' not found in the 'images' directory with extensions .jpg, .jpeg, or .png")
                # Use a fallback or placeholder image
                html_content = html_content.replace("{{imageholder}}", "../images/placeholder.jpg")

        # Apply parameters from animations.txt first, then from the input file task
        all_params = {**anim_data.get('data', {}), **task.get('params', {})}
        for key, value in all_params.items():
            html_content = html_content.replace(f"{{{{{key}}}}}", value)
        
        # Replace video placeholder if provided
        if video_source:
            html_content = html_content.replace("{{videoholder}}", video_source)
            print(f"  - Replaced {{videoholder}} with: {video_source}")
            
        render_html_path = os.path.join(TEMP_FOLDER, f"render_{task['id']}.html")
        with open(render_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        for filename in os.listdir(template_path):
            source_file = os.path.join(template_path, filename)
            if os.path.isfile(source_file):
                shutil.copy(source_file, TEMP_FOLDER)
    except Exception as e:
        print(f"  ERROR: Failed to prepare HTML template: {e}")
        return None

    output_video_path = os.path.join(TEMP_FOLDER, f"render_capture_{task['id']}.mp4")

    try:
        with sync_playwright() as p:
            launch_args = [
                '--autoplay-policy=no-user-gesture-required',
                '--allow-file-access-from-files',
            ]
            
            # --- THIS IS THE FIX ---
            # Use the system's installed Chrome (which has all media codecs)
            # instead of Playwright's default open-source Chromium.
            browser = p.chromium.launch(
                headless=True,
                args=launch_args,
                channel="chrome"  # <-- Tells Playwright to find your real Chrome browser
            )
            # -----------------------
            
            context = browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                record_video_dir=TEMP_FOLDER,
                record_video_size={'width': 1920, 'height': 1080}
            )
            page = context.new_page()
            
            page.goto(pathlib.Path(os.path.abspath(render_html_path)).as_uri())
            print("  - Browser started, recording with Playwright...")
            page.wait_for_timeout(1000) # Wait a bit for the page to load
            
            page.wait_for_timeout(task['duration'] * 1000)
            
            video_temp_path = page.video.path()
            context.close()
            browser.close()

        shutil.move(video_temp_path, output_video_path)

    except Exception as e:
        print(f"  ERROR: An error occurred during Playwright automation: {e}")
        return None
    
    if os.path.exists(output_video_path):
        print(f"  SUCCESS: Animation rendered to {output_video_path}")
        return {"id": f"html_{task['id']}_{anim_id}", "source_path": output_video_path, "type": "rendered_html", "duration": task['duration']}
    else:
        print("  FAILURE: Output video file was not created by Playwright.")
        return None

def process_and_standardize_clip(task_id, footage_info, duration):
    input_path = footage_info['source_path']
    final_clip_path = os.path.join(TEMP_FOLDER, f"clip_{task_id}.ts")
    print(f"    - Standardizing clip {task_id} to TS format...")
    
    vf_filter = f"scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1,fps={STANDARDIZED_FRAMERATE}"
    
    cmd = [FFMPEG_PATH, '-y']
    
    is_video = footage_info['type'] not in ['local_image', 'downloaded_image']
    loop_input = False

    if is_video:
        actual_duration = footage_info.get('duration', get_video_duration(input_path))
        if actual_duration > 0 and actual_duration < duration:
            print(f"    - Input video is shorter ({actual_duration}s) than required ({duration}s). Looping video.")
            loop_input = True
            cmd.extend(['-stream_loop', '-1'])
    else: # is image
        cmd.extend(['-loop', '1'])

    start_time = footage_info.get('start', 0)
    cmd.extend(['-ss', str(start_time), '-i', input_path])

    # Add silent audio and process
    cmd.extend([
        '-f', 'lavfi', '-i', f'anullsrc=channel_layout=stereo:sample_rate=48000:d={duration}',
        '-t', str(duration),
        '-vf', vf_filter,
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-c:v', 'mpeg2video', '-q:v', '2',
        '-c:a', 'mp2', '-b:a', '192k',
        '-shortest',
        final_clip_path
    ])

    run_ffmpeg_command(cmd)
    return final_clip_path

def create_placeholder_clip(task, duration):
    final_clip_path = os.path.join(TEMP_FOLDER, f"clip_{task['id']}.ts")
    
    if task.get('animation_id'):
        raw_text = f"Animation '{task['animation_id']}' failed to render."
    else:
        all_terms = ", ".join(task.get('search_terms', ['N/A']))
        raw_text = f"Footage not found for: {all_terms}"

    escaped_text = raw_text.replace("'", r"\'" ).replace(":", r"\:")
    vf_filter = f"drawtext=text='{escaped_text}':fontcolor=white:fontsize=40:x=(w-text_w)/2:y=(h-text_h)/2"
    
    cmd = [FFMPEG_PATH, '-y', '-f', 'lavfi', '-i', f'color=c=black:s={STANDARDIZED_RESOLUTION}:d={duration}:r={STANDARDIZED_FRAMERATE}', '-vf', vf_filter, '-c:v', 'mpeg2video', '-q:v', '2', '-an', final_clip_path]
    run_ffmpeg_command(cmd)
    return final_clip_path

def concatenate_and_encode(clip_paths, output_file, encoder):
    list_file_path = os.path.join(TEMP_FOLDER, "concat_list.txt")
    with open(list_file_path, 'w') as f:
        for path in clip_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")
    
    cmd = [FFMPEG_PATH, '-y', '-f', 'concat', '-safe', '0', '-i', list_file_path, '-c:v', encoder, '-preset', 'fast', '-crf', '23', '-c:a', 'aac', '-b:a', '192k', '-pix_fmt', 'yuv420p', output_file]
    print(f"\n-> Joining all clips and encoding final video to {output_file} using {encoder}...")
    run_ffmpeg_command(cmd)

def group_nogreenscreen_tasks(tasks):
    """Group consecutive nogreenscreen tasks with the same template style."""
    groups = []
    current_group = None
    
    for task in tasks:
        if task.get('nogreenscreen') and task.get('template_style'):
            template_style = task['template_style']
            
            # If this is the first nogreenscreen task or different template, start new group
            if (current_group is None or 
                current_group['template_style'] != template_style or
                current_group['end_time'] != task['start_time']):
                
                if current_group:
                    groups.append(current_group)
                
                current_group = {
                    'template_style': template_style,
                    'start_time': task['start_time'],
                    'end_time': task['end_time'],
                    'tasks': [task],
                    'group_id': f"nogreenscreen_group_{len(groups)}"
                }
            else:
                # Extend current group
                current_group['end_time'] = task['end_time']
                current_group['tasks'].append(task)
        else:
            # Non-nogreenscreen task, close current group if exists
            if current_group:
                groups.append(current_group)
                current_group = None
    
    # Don't forget the last group
    if current_group:
        groups.append(current_group)
    
    return groups

def create_video_source_with_time(output_filename, start_time, end_time):
    """Create video source string with time fragment for HTML templates."""
    return f"../{output_filename}#t={start_time},{end_time}"

def main():
    best_encoder = get_best_encoder()
    animations_db = parse_animations_file(ANIMATIONS_FILE)
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        return
    txt_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.txt')]
    if not txt_files:
        print(f"No .txt files found in the '{INPUT_FOLDER}' folder.")
        return

    print(f"\nFound {len(txt_files)} file(s) to process: {', '.join(txt_files)}")

    for filename in txt_files:
        base_name = os.path.splitext(filename)[0]
        final_output_file = f"{base_name}.mp4"
        error_report_path = f"{base_name}_error_report.txt"

        try:
            print(f"\n{'='*20}\nProcessing: {filename}\n{'='*20}")
            if os.path.exists(TEMP_FOLDER):
                try:
                    shutil.rmtree(TEMP_FOLDER)
                except PermissionError:
                    print(f"Warning: Could not remove temp folder, trying again...")
                    time.sleep(1)
                    try:
                        shutil.rmtree(TEMP_FOLDER)
                    except PermissionError:
                        print(f"Warning: Temp folder still in use, continuing...")
            if not os.path.exists(TEMP_FOLDER):
                os.makedirs(TEMP_FOLDER)
            
            input_filepath = os.path.join(INPUT_FOLDER, filename)
            tasks = parse_input_file(input_filepath)
            if not tasks:
                print(f"No valid tasks found in {filename}. Skipping.")
                continue

            # --- STAGE 1: Initial Processing and First-Pass Video Generation ---
            print(f"\n--- STAGE 1: Processing all tasks to create a base video ---")

            # --- Step 1.1: Pre-render global overlays ---
            overlay_groups = {}
            for task in tasks:
                if task.get("overlay"):
                    overlay_details = task["overlay"]
                    # Create a shorter, safer group key using hash to avoid long filenames
                    import hashlib
                    text_content = overlay_details.get('text_content', '')
                    text_hash = hashlib.md5(text_content.encode()).hexdigest()[:8]
                    group_key = f"{overlay_details.get('animation','')}_{text_hash}_{overlay_details.get('font','')}_{overlay_details.get('color','')}"
                    if group_key not in overlay_groups:
                        overlay_groups[group_key] = {
                            "global_start": task["start_time"],
                            "global_end": task["end_time"],
                            "overlay_details": overlay_details,
                            "tasks": [task]
                        }
                    else:
                        group = overlay_groups[group_key]
                        group['global_start'] = min(group['global_start'], task['start_time'])
                        group['global_end'] = max(group['global_end'], task['end_time'])
                        group['tasks'].append(task)

            for group_key, group in overlay_groups.items():
                duration = group["global_end"] - group["global_start"]
                print(f"\n-> Rendering global overlay for group '{group_key}' covering {group['global_start']} to {group['global_end']} seconds (duration: {duration}s)...")
                rep_task = group["tasks"][0]
                overlay_clip = render_overlay(rep_task, animations_db, best_encoder, duration_override=duration, group_id=group_key)
                group["overlay_file"] = overlay_clip

            # --- Step 1.2: Process each task into a standardized .ts clip ---
            first_pass_clips = []
            for task in tasks:
                # For nogreenscreen tasks, we treat them as regular video/image tasks for this pass
                if task.get("overlay"):
                    overlay_details = task["overlay"]
                    # Create the same shorter, safer group key using hash
                    import hashlib
                    text_content = overlay_details.get('text_content', '')
                    text_hash = hashlib.md5(text_content.encode()).hexdigest()[:8]
                    group_key = f"{overlay_details.get('animation','')}_{text_hash}_{overlay_details.get('font','')}_{overlay_details.get('color','')}"
                    group = overlay_groups.get(group_key, {})
                    overlay_clip = group.get("overlay_file")
                    if not overlay_clip:
                        print(f"  ERROR: Overlay clip not found for task {task['id']}. Processing without overlay.")
                        footage_info = find_footage(task)
                        clip_path = process_and_standardize_clip(task['id'], footage_info, task['duration']) if footage_info else create_placeholder_clip(task, task['duration'])
                    else:
                        print(f"  - Processing task {task['id']} with overlay...")
                        footage_info = find_footage(task)
                        clip_path = process_and_standardize_clip(task['id'], footage_info, task['duration']) if footage_info else create_placeholder_clip(task, task['duration'])
                        overlay_offset = task["start_time"] - group["global_start"]
                        overlayed_clip = os.path.join(TEMP_FOLDER, f"overlayed_{task['id']}.ts")
                        apply_overlay(clip_path, overlay_clip, overlayed_clip, overlay_offset=overlay_offset)
                        clip_path = overlayed_clip
                else: # Regular video, image, imgvideo, or html task
                    if task['media_type'] in ['video', 'image', 'imgvideo']:
                        footage_info = find_footage(task)
                    elif task['media_type'] == 'html':
                        footage_info = render_html_animation(task, animations_db)
                    else:
                        footage_info = None
                    clip_path = process_and_standardize_clip(task['id'], footage_info, task['duration']) if footage_info else create_placeholder_clip(task, task['duration'])
                
                first_pass_clips.append({
                    "task": task,
                    "path": clip_path,
                    "duration_frames": int(task['duration'] * STANDARDIZED_FRAMERATE)
                })

            # --- Step 1.3: Concatenate first-pass clips into a base video ---
            first_pass_video_path = f"{base_name}_pass1.mp4"
            ts_clip_paths = [item['path'] for item in first_pass_clips]
            concatenate_and_encode(ts_clip_paths, first_pass_video_path, best_encoder)

            # --- STAGE 2: Render nogreenscreen effects and perform final assembly ---
            print(f"\n--- STAGE 2: Rendering 'nogreenscreen' effects using the base video ---")
            final_clips_details = []
            nogreenscreen_tasks = [item for item in first_pass_clips if item['task'].get('nogreenscreen')]

            if not nogreenscreen_tasks:
                print("-> No 'nogreenscreen' tasks found. The first pass video is the final video.")
                # Remove existing output file if it exists before renaming
                if os.path.exists(final_output_file):
                    os.remove(final_output_file)
                os.rename(first_pass_video_path, final_output_file)
                final_clips_details = first_pass_clips # Use first pass clips for XML
            else:
                # --- Step 2.1: Render new clips for nogreenscreen tasks ---
                rendered_nogreenscreen_clips = {}
                for item in nogreenscreen_tasks:
                    task = item['task']
                    print(f"\n-> Rendering HTML animation for nogreenscreen Task {task['id']}' with Playwright")
                    video_source = f"../{first_pass_video_path}#t={task['start_time']},{task['end_time']}"
                    
                    fake_task = {
                        'id': f"{task['id']}_nogreenscreen",
                        'animation_id': task['template_style'],
                        'duration': task['duration'],
                        'params': task.get('params', {}) # Pass parameters to the renderer
                    }
                    
                    footage_info = render_html_animation(fake_task, animations_db, video_source)
                    if footage_info:
                        clip_path = process_and_standardize_clip(fake_task['id'], footage_info, task['duration'])
                        rendered_nogreenscreen_clips[task['id']] = clip_path
                        print(f"  SUCCESS: Rendered nogreenscreen clip for task {task['id']}")
                    else:
                        print(f"  ERROR: Failed to render nogreenscreen clip for task {task['id']}. Using placeholder.")
                        rendered_nogreenscreen_clips[task['id']] = create_placeholder_clip(task, task['duration'])

                # --- Step 2.2: Assemble the final list of clips, replacing placeholders ---
                final_ts_paths = []
                for item in first_pass_clips:
                    task_id = item['task']['id']
                    if task_id in rendered_nogreenscreen_clips:
                        # Replace the original clip with the new nogreenscreen version
                        new_path = rendered_nogreenscreen_clips[task_id]
                        final_ts_paths.append(new_path)
                        item['path'] = new_path # Update for XML generation
                    else:
                        # Keep the original clip
                        final_ts_paths.append(item['path'])
                
                final_clips_details = first_pass_clips
                concatenate_and_encode(final_ts_paths, final_output_file, best_encoder)
                os.remove(first_pass_video_path) # Clean up the intermediate video

            # --- Final encoding and XML generation (clips are kept in the original input order) ---
            if final_clips_details:
                print(f"\nSuccess! Video saved to {final_output_file}")
            else:
                print(f"\nNo clips were processed for {filename}. No final video created.")

        except Exception as e:
            with open(error_report_path, "w") as f:
                f.write(f"An error occurred while processing {filename}:\n")
                f.write(str(e))
                f.write(traceback.format_exc())
            print(f"An error occurred. A report has been generated at {error_report_path}")


if __name__ == "__main__":
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
    if not os.path.exists(TEMPLATES_FOLDER):
        os.makedirs(TEMPLATES_FOLDER)
    main()
