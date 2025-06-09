import streamlit as st
import os
import subprocess
import time
from streamlit_autorefresh import st_autorefresh
import sys
sys.path.append('/Users/michaelrice/Documents/GitHub/Thesis/MSc_AI_Thesis/notebooks_and_scripts')
from vrs_extractor import VRSDataExtractor
from data_processing_main import DataProcessor
import json
from llama_cpp import Llama

st.set_page_config(page_title="Driver Scoring Dashboard", layout="centered")

st.title("🚗 Driver Scoring Dashboard")

root_dir = "/Users/michaelrice/Documents/GitHub/Thesis/MSc_AI_Thesis/data"
session_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

with st.spinner("⏳ Loading LLM..."):
    llm = Llama(model_path="/Users/michaelrice/Documents/GitHub/Thesis/MSc_AI_Thesis/utilities/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    n_ctx=2048,        
    n_threads=8,      
    n_batch=64,         
    use_mlock=True,    
    verbose=True        
)

res = {}

def check_files_exist(files):
    for file in files:
        if not os.path.exists(file):
            return False
    return True

def frame_progress_callback(current, total):
    percent = int((current / total) * 100)
    frame_progress.progress(percent, text=f"Frame Extraction: {percent}%")
    frame_progress_status.text(f"Extracted {current}/{total} frames")

def object_detection_progress(current, total):
    percent = int((current / total) * 100)
    obj_det_progress.progress(percent, text=f"Object Detection: {percent}%")
    obj_det_status.text(f"Processed {current}/{total} frames")

def driver_evaluation_progress(current, total):
    percent = int((current / total) * 100)
    dr_eval_progress.progress(percent, text=f"Driver Evaluation: {percent}%")
    dr_eval_status.text(f"Evaluated {current}/{total} frames")

def find_vrs_file(folder_path):
    for fname in os.listdir(folder_path):
        if fname.endswith(".vrs"):
            return fname  # Return the first .vrs file found
    return None

def check_output_folder(folder_path, base_name):
    mps = f'mps_{base_name}_vrs'
    output_folder = os.path.join(folder_path, mps)
    return os.path.exists(output_folder) and os.path.isdir(output_folder)

# --- Sidebar for folder selection ---
with st.sidebar:
    st.header("Session Selection")
    selected_folder = st.selectbox("Select a session folder:", session_folders)
    st.info("Select a session folder to process driving data and score the driver.")


    info_file_path = find_vrs_file(os.path.join(root_dir, selected_folder))
    if info_file_path is not None:  # filename without extension
        base_name = os.path.splitext(info_file_path)[0]
        info_file_path2 = os.path.join(root_dir, selected_folder, f'{base_name}.vrs.json')
        if os.path.exists(info_file_path2):
            with open(info_file_path2, 'r') as f:
                info_data = json.load(f)
    

        st.header('Data Healthcheck and Information')

        with st.expander("Data Information"):
            data_profile_info = info_data['custom_profile']['description'][8:-1]
            image_format = info_data['custom_profile']['rgb_camera']['image_format']
            num_frames = info_data['data_quality_stats']['rgb_camera']['processed']
            st.write(f'{data_profile_info}')
            st.write(f"**Number of Frames:** {num_frames}")
            st.write(f"**Image Format:** {image_format}")

        with st.expander("Recording Healthcheck"):
            rgb_score = info_data['data_quality_stats']['rgb_camera']['score']
            st.write(f"**RGB Camera Score:** {rgb_score}")
            imu1_score = info_data['data_quality_stats']['imu_1']['score']
            st.write(f"**IMU 1 Score:** {imu1_score}")
            imu2_score = info_data['data_quality_stats']['imu_2']['score']
            st.write(f"**IMU 2 Score:** {imu2_score}")
            et_camera_score = info_data['data_quality_stats']['et_camera']['score']
            st.write(f"**Eye Tracking Camera Score:** {et_camera_score}")
            s1_score = info_data['data_quality_stats']['slam_camera_1']['score']
            st.write(f"**SLAM Camera 1 Score:** {s1_score}")
            s2_score = info_data['data_quality_stats']['slam_camera_2']['score']
            st.write(f"**SLAM Camera 2 Score:** {s2_score}")



# Initialize session state variables
if "processing" not in st.session_state:
    st.session_state.processing = False
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "stop_waiting" not in st.session_state:
    st.session_state.stop_waiting = False
if "process_finished" not in st.session_state:
    st.session_state.process_finished = False
if "base_name" not in st.session_state:
    st.session_state.base_name = None
if "result" not in st.session_state:
    st.session_state.result = None
if "results_dict" not in st.session_state:
    st.session_state.results_dict = {}
if 'review_mode' not in st.session_state:
    st.session_state.review_mode = False
if "tips_output" not in st.session_state:
    st.session_state.tips_output = None
if "preview_mode" not in st.session_state:
    st.session_state.preview_mode = False
if "vde" not in st.session_state:
    st.session_state.vde = None
if "llm_mode" not in st.session_state:
    st.session_state.llm_mode = False
if 'selected_mistake' not in st.session_state:
    st.session_state.selected_mistake = None



folder_path = os.path.join(root_dir, selected_folder)
vrs_file = find_vrs_file(folder_path)
base_name = os.path.splitext(vrs_file)[0]  # filename without extension

files_to_check = [os.path.join(folder_path, f'mps_{base_name}_vrs', 'eye_gaze', 'general_eye_gaze.csv'),
                #   os.path.join(folder_path, f'mps_{base_name}_vrs', 'eye_gaze', 'personalized_eye_gaze.csv'),
                  os.path.join(folder_path, f'mps_{base_name}_vrs', 'hand_tracking', 'hand_tracking_results.csv')]

st.markdown("---")

frame_progress = st.progress(0, text="Frame Extraction")
frame_progress_status = st.empty()

obj_det_progress = st.progress(0, text="Object Detection")
obj_det_status = st.empty()

dr_eval_progress = st.progress(0, text="Driver Evaluation")
dr_eval_status = st.empty()

# Use columns for buttons side by side
col1, col2 , col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

with col1:
    process_btn = st.button("▶️ Process Driving Data", disabled=st.session_state.processing)

with col2:
    preview_btn = st.button("👀 Preview Data", disabled=st.session_state.processing)
    
with col3:
    score_btn = st.button("🏆 Session Scores", disabled=st.session_state.processing)

with col4:
    review_btn = st.button("📋 Review Mistakes", disabled=st.session_state.processing)
    
with col5:
    tips_btn = st.button("💡 Tips for Improvement", disabled=st.session_state.processing)




if process_btn and not st.session_state.processing:
    st.session_state.review_mode = False  
    st.session_state.preview_mode = False
    st.session_state.llm_mode = False

    vrs_file = find_vrs_file(folder_path)

    if vrs_file:
        st.info(f"🗂️ Processing file: **{vrs_file}**")

        base_name = os.path.splitext(vrs_file)[0]  # filename without extension

        input_dir = os.path.expanduser(folder_path)
        cmd = ["aria_mps", "single", "-i", input_dir]
        full_vrs_path = os.path.join(input_dir, vrs_file)
        st.session_state.vde = VRSDataExtractor(full_vrs_path)
        dp = DataProcessor(st.session_state.vde)
        st.session_state.results_dict = dp.vrs_processing(full_vrs_path, callbacks={
        "object_detection": object_detection_progress,
        "image_extraction": frame_progress_callback,
        "driving_evaluation": driver_evaluation_progress
        }) 
        if check_files_exist(files_to_check):
            st.success("✅ All files already processed!")       
        else:
            st.info("🔄 Starting processing with aria_mps...")
            try:
                subprocess.Popen(cmd)  # non-blocking
                st.session_state.processing = True
                st.session_state.start_time = time.time()
                st.session_state.stop_waiting = False
                st.session_state.process_finished = False
                st.session_state.base_name = base_name  # save base name for output check
            except Exception as e:
                st.error(f"❌ Failed to run aria_mps: {e}")
    else:
        st.warning("⚠️ No .vrs files found in the selected folder.")

if preview_btn:
    st.session_state.review_mode = False 
    st.session_state.llm_mode = False
    st.session_state.preview_mode = True

    vid = open(st.session_state.results_dict['video_path'], 'rb')
    video_bytes = vid.read()

    if st.session_state.results_dict is None or not st.session_state.results_dict:
        st.warning("⚠️ No VRS file found to preview.")
    else:
        st.video(video_bytes)



if score_btn:
    if not st.session_state.results_dict or 'scores' not in st.session_state.results_dict:
        st.warning("⚠️ No scores available. Please process the data first.")
    else:
        st.session_state.review_mode = False
        st.session_state.preview_mode = False
        st.session_state.llm_mode = False

        with st.spinner("🏅 Session Results"):
            time.sleep(2)  # Simulate processing time

        st.success("🎉 Done!")
        st.write(f"**Score for session {selected_folder}:** {st.session_state.results_dict['scores'][0]:.2f}%")
        st.write(f"**Left Mirror Checks:** {st.session_state.results_dict['scores'][1]:.2f}%")
        st.write(f"**Right Mirror Checks:** {st.session_state.results_dict['scores'][2]:.2f}%")
        st.write(f"**Rearview Mirror Checks:** {st.session_state.results_dict['scores'][3]:.2f}%")
        st.write(f"**Mobile Phone Used ** {st.session_state.results_dict['scores'][4]} times")



if tips_btn:
    if not st.session_state.results_dict or 'scores' not in st.session_state.results_dict:
        st.warning("⚠️ No scores available. Please process the data first.")
    else:
        st.session_state.processing = False
        st.session_state.review_mode = False
        st.session_state.preview_mode = False
        st.session_state.llm_mode = True

        overall_score = st.session_state.results_dict['scores'][0]
        left_mirror_score = st.session_state.results_dict['scores'][1]
        right_mirror_score = st.session_state.results_dict['scores'][2]
        rearview_mirror_score = st.session_state.results_dict['scores'][3]
        mobile_phone_count = st.session_state.results_dict['scores'][4]

        prompt = (
        f"""You are an expert driving instructor. Based on the session data below, "
        identify which mirror-check behaviors are weak (below 90%) and give 3 specific tips "
        only to improve those weak areas:\n"
        - Left Wing Mirror Checks Every 30s: {left_mirror_score}\n"
        - Right Wing Mirror Checks Every 30s: {right_mirror_score}\n"
        - Rearview Mirror Checks Every 30s: {rearview_mirror_score}\n"
        - Overall Score: {overall_score}\n"
        Focus only on the weakest area. Provide actionable, specific, concise tips """)

        with st.spinner("🧠 Generating tips..."):
            output = llm(prompt, max_tokens=300)
            st.session_state.tips_output = output["choices"][0]["text"].strip()
    
    

if review_btn:
    if not st.session_state.results_dict or 'mistake_sections' not in st.session_state.results_dict:
        st.warning("⚠️ No mistakes found in the session. Please process the data first.")
    else:
        st.session_state.preview_mode = False
        st.session_state.llm_mode = False
        st.session_state.review_mode = True

if st.session_state.review_mode:
    mistake_video_paths = st.session_state.results_dict.get('mistake_video_paths', [])

    if not mistake_video_paths:
        st.warning("⚠️ No mistakes found in this session.")
    else:
        mistakes = [m.split('/')[-1].split('.')[0] for m in mistake_video_paths]

        # Initialize selected mistake if not already set or if it's not valid
        if ('selected_mistake_label' not in st.session_state or 
            st.session_state.selected_mistake_label not in mistakes):
            st.session_state.selected_mistake_label = mistakes[0]

        # Create the selectbox with a unique key and proper state management
        current_selection = st.selectbox(
            "Select a mistake to review:",
            options=mistakes,
            index=mistakes.index(st.session_state.selected_mistake_label),
            key="mistake_selector"  # Add unique key
        )

        # Update session state only if selection actually changed
        if current_selection != st.session_state.selected_mistake_label:
            st.session_state.selected_mistake_label = current_selection

        # Find and display the selected video
        selected_video_path = None
        for path in mistake_video_paths:
            if st.session_state.selected_mistake_label in path:
                selected_video_path = path
                break

        if selected_video_path and os.path.exists(selected_video_path):
            try:
                # Add some spacing and a container for better layout
                st.markdown("---")
                st.subheader(f"Reviewing: {st.session_state.selected_mistake_label}")
                
                # Show loading message for large videos
                with st.spinner("Loading video..."):
                    with open(selected_video_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                
                # Use a container to ensure consistent layout
                video_container = st.container()
                with video_container:
                    st.video(video_bytes)
                    
            except Exception as e:
                st.error(f"❌ Error loading video: {e}")
        else:
            st.error("❌ Selected video file not found.")







# Auto-refresh every 30 seconds (30000 milliseconds)
if not st.session_state.processing and not st.session_state.review_mode and not st.session_state.preview_mode and not st.session_state.llm_mode:
    st_autorefresh(interval=30000, key="datarefresh")




if st.session_state.processing and not st.session_state.process_finished:
    elapsed = int(time.time() - st.session_state.start_time)
    st.info(f"⏳ Waiting for output files... elapsed time: {elapsed // 60}m {elapsed % 60}s")

    if check_files_exist(files_to_check):
        st.success("✅ Processing Completed!")
        st.session_state.processing = False
        st.session_state.process_finished = True

    elif st.button("⏹️ Stop Waiting"):
        st.session_state.stop_waiting = True
        st.session_state.processing = False
        st.warning("⚠️ Stopped waiting for output files.")

    elif elapsed > 100 * 60:  # Timeout after 100 minutes
        st.error("⏰ Timeout: Output files not found after 20 minutes.")
        st.session_state.processing = False

if st.session_state.process_finished:
    st.success("You can now score the driver.")





if st.session_state.tips_output:
    st.info("💡 Tips for driving improvement:")
    st.write(f"**Tips for improvement:** {st.session_state.tips_output}")



# Add some bottom padding
st.markdown("<br><br>", unsafe_allow_html=True)
