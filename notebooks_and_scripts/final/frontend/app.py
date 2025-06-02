import streamlit as st
import os
import subprocess
import time
from streamlit_autorefresh import st_autorefresh
import sys
sys.path.append('/Users/michaelrice/Documents/GitHub/Thesis/MSc_AI_Thesis/notebooks_and_scripts')
from vrs_extractor import VRSDataExtractor
from data_processing_main import DataProcessor
st.set_page_config(page_title="Driver Scoring Dashboard", layout="centered")

st.title("🚗 Driver Scoring Dashboard")

root_dir = "/Users/michaelrice/Documents/GitHub/Thesis/MSc_AI_Thesis/data"

session_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]



def frame_progress_callback(current, total):
    percent = int((current / total) * 100)
    frame_progress.progress(percent, text=f"Frame Extraction: {percent}%")
    frame_progress_status.text(f"Extracted {current}/{total} frames")

def object_detection_progress(current, total):
    percent = int((current / total) * 100)
    obj_det_progress.progress(percent, text=f"Object Detection: {percent}%")
    obj_det_status.text(f"Processed {current}/{total} frames")

# --- Sidebar for folder selection ---
with st.sidebar:
    st.header("Session Selection")
    selected_folder = st.selectbox("Select a session folder:", session_folders)
    st.info("Select a session folder to process driving data and score the driver.")

    st.header('Data Information')
    st.write(f'**Data Modalities:** RGB, Gaze, Hand Landmarking, IMU, Object Detections, GPS')

    len_frames = 1000
    st.write(f'**Recording Length:** {len_frames} frames / {(len_frames / 15) / 60:.2f} mins')


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

def find_vrs_file(folder_path):
    for fname in os.listdir(folder_path):
        if fname.endswith(".vrs"):
            return fname  # Return the first .vrs file found
    return None

def check_output_folder(folder_path, base_name):
    mps = f'mps_{base_name}_vrs'
    output_folder = os.path.join(folder_path, mps)
    return os.path.exists(output_folder) and os.path.isdir(output_folder)

folder_path = os.path.join(root_dir, selected_folder)

st.markdown("---")

# Use columns for buttons side by side
col1, col2 , col3 = st.columns([1, 1, 1])

with col1:
    process_btn = st.button("▶️ Process Driving Data", disabled=st.session_state.processing)

with col2:
    score_btn = st.button("🏆 Score Driver")

with col3:
    preview_btn = st.button("👀 Preview Data", disabled=st.session_state.processing)

frame_progress = st.progress(0, text="Frame Extraction")
frame_progress_status = st.empty()

obj_det_progress = st.progress(0, text="Object Detection")
obj_det_status = st.empty()

if process_btn and not st.session_state.processing:
    vrs_file = find_vrs_file(folder_path)

    if vrs_file:
        st.info(f"🗂️ Processing file: **{vrs_file}**")

        base_name = os.path.splitext(vrs_file)[0]  # filename without extension

        input_dir = os.path.expanduser(folder_path)
        cmd = ["aria_mps", "single", "-i", input_dir]
        full_vrs_path = os.path.join(input_dir, vrs_file)
        vde = VRSDataExtractor(full_vrs_path)
        dp = DataProcessor(vde)
        res = dp.vrs_processing(full_vrs_path, callbacks={
        "object_detection": object_detection_progress,
        "image_extraction": frame_progress_callback})        
        
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

# Auto-refresh every 30 seconds (30000 milliseconds)
st_autorefresh(interval=30000, key="datarefresh")

if st.session_state.processing and not st.session_state.process_finished:
    elapsed = int(time.time() - st.session_state.start_time)
    st.info(f"⏳ Waiting for output files... elapsed time: {elapsed // 60}m {elapsed % 60}s")

    if check_output_folder(folder_path, st.session_state.base_name):
        st.success("✅ Processing complete!")
        st.session_state.processing = False
        st.session_state.process_finished = True

    elif st.button("⏹️ Stop Waiting"):
        st.session_state.stop_waiting = True
        st.session_state.processing = False
        st.warning("⚠️ Stopped waiting for output files.")

    elif elapsed > 20 * 60:  # Timeout after 20 minutes
        st.error("⏰ Timeout: Output files not found after 20 minutes.")
        st.session_state.processing = False

if st.session_state.process_finished:
    st.success("You can now score the driver.")

if preview_btn:
    st.info("👀 Previewing data...")


    st.write(f"**Previewing data for session:** {vrs_file}")



if score_btn:
    with st.spinner("🏅 Scoring driver..."):
        # Replace with your actual scoring logic
        time.sleep(2)
        st.session_state.result = 10
    st.success("🎉 Done!")
    st.write(f"**Score for session {selected_folder}:** {st.session_state.result}")

    llm_prompt = ''
    llm_result = ''
    st.write(f'Tips for driving improvement : {llm_result}')

# Add some bottom padding
st.markdown("<br><br>", unsafe_allow_html=True)
