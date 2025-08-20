import streamlit as st
import os
import subprocess
import time
# from streamlit_autorefresh import st_autorefresh
import sys
sys.path.append('MSc_AI_Thesis/notebooks_and_scripts')
from dataset_scripts.vrs_extractor import VRSDataExtractor
from dataset_scripts.DataExtractionMain import DataProcessor
import json
from llama_cpp import Llama



st.set_page_config(page_title="Driver Scoring Dashboard", layout="centered")

st.title("🚗 Driver Scoring Dashboard")

root_dir = "MSc_AI_Thesis/notebooks_and_scripts/dashboard/test_data"
session_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

# Loading the local LLM
with st.spinner("⏳ Loading LLM..."):
    llm = Llama(model_path="MSc_AI_Thesis/utilities/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    n_ctx=2048,        
    n_threads=8,      
    n_batch=64,         
    use_mlock=True,    
    verbose=True        
)

res = {}

def check_files_exist(files):

    """ Check if all files in the list exist."""

    for file in files:
        if not os.path.exists(file):
            return False
    return True

def generate_driving_feedback_prompt(results_dict, mirror_scores, overall_score):

    """ Generate a prompt for the local LLM based on the results dictionary and mirror scores."""

    weak_areas = []
    
    left_mirror_score = mirror_scores.get('left', 0) * 100
    right_mirror_score = mirror_scores.get('right', 0) * 100  
    rearview_mirror_score = mirror_scores.get('rearview', 0) * 100
    
    if left_mirror_score < 90:
        weak_areas.append(f"Left Wing Mirror Checks: {left_mirror_score:.1f}%")
    if right_mirror_score < 90:
        weak_areas.append(f"Right Wing Mirror Checks: {right_mirror_score:.1f}%")
    if rearview_mirror_score < 90:
        weak_areas.append(f"Rearview Mirror Checks: {rearview_mirror_score:.1f}%")
    
    # Add behavioral concerns
    if results_dict['one_handed_percent'] > 5:
        weak_areas.append(f"One-Handed Driving: {results_dict['one_handed_percent']:.1f}%")
    if results_dict['crossover_occurrences'] > 2:
        weak_areas.append(f"Crossover Steering Occurrences: {results_dict['crossover_occurrences']} times")
    if results_dict['infotainment_distraction_count'] > 3:
        weak_areas.append(f"Infotainment Distraction Occurrences: {results_dict['infotainment_distraction_count']:.1f}%")
    
    # Add gaze analysis
    gaze_duration = results_dict.get("mean_gaze_fixation_duration_sec", 0)
    if gaze_duration > 3:
        weak_areas.append(f"Excessive Gaze Fixation: {gaze_duration:.1f} seconds (should be less than 3.0s)")
    elif gaze_duration < 0.5:
        weak_areas.append(f"Insufficient Visual Scanning: {gaze_duration:.1f} seconds (should be >0.5s)")
    
    speedometer_checks = results_dict.get('speedometer_checks', 0)
    session_duration_minutes = len(results_dict.get('overlays', [])) / (15 * 60)  # Assuming 15fps
    expected_speed_checks = max(1, int(session_duration_minutes / 2))  # Every 2 minutes minimum
    
    if speedometer_checks < expected_speed_checks:
        weak_areas.append(f"Speedometer Checks: {speedometer_checks} times ({expected_speed_checks} expected)")

    prompt = f"""DRIVING SESSION ANALYSIS

    PERFORMANCE SUMMARY:
    • Overall Score: {overall_score:.1f}%
    • Session Duration: ~{session_duration_minutes:.1f} minutes
    • Total Behaviors Analyzed: 8 categories

    CURRENT PERFORMANCE LEVELS:
    ✓ Left Wing Mirror: {left_mirror_score:.1f}% (Target: 90%+)
    ✓ Right Wing Mirror: {right_mirror_score:.1f}% (Target: 90%+)  
    ✓ Rearview Mirror: {rearview_mirror_score:.1f}% (Target: 90%+)
    ✓ One-Handed Driving: {results_dict['one_handed_percent']:.1f}% (Target: <5%)
    ✓ Crossover Steering: {results_dict['crossover_occurrences']:.1f}% (Target: <2%)
    ✓ Infotainment Distraction: {results_dict['infotainment_distraction_count']:.1f}% (Target: <3%)
    ✓ Speedometer Monitoring: {speedometer_checks} checks (Target: {expected_speed_checks}+)
    ✓ Gaze Focus Duration: {gaze_duration:.1f}s (Target: 0.5-2.0s)

    AREAS REQUIRING ATTENTION ({len(weak_areas)} identified):
    {chr(10).join(f"• {area}" for area in weak_areas) if weak_areas else "• All areas performing well!"}

    TASK: Provide specific improvement strategies for the identified weak areas only. Focus on immediate, actionable techniques the driver can practice."""

    return prompt

def frame_progress_callback(current, total):

    """ Update the frame extraction progress bar and status text."""

    percent = int((current / total) * 100)
    frame_progress.progress(percent, text=f"Frame Extraction: {percent}%")
    frame_progress_status.text(f"Extracted {current}/{total} frames")

def object_detection_progress(current, total):

    """ Update the object detection progress bar and status text."""

    percent = int((current / total) * 100)
    obj_det_progress.progress(percent, text=f"Object Detection: {percent}%")
    obj_det_status.text(f"Processed {current}/{total} frames")

def driver_evaluation_progress(current, total):

    """ Update the driver evaluation progress bar and status text."""

    percent = int((current / total) * 100)
    dr_eval_progress.progress(percent, text=f"Driver Evaluation: {percent}%")
    dr_eval_status.text(f"Evaluated {current}/{total} frames")

def find_vrs_file(folder_path):

    """ Find the first .vrs file in the specified folder."""

    for fname in os.listdir(folder_path):
        if fname.endswith(".vrs") and not fname.startswith("._"):
            return fname  # Return the first .vrs file found
    return None

def check_output_folder(folder_path, base_name):

    """ Check if the output folder for the processed data exists."""

    mps = f'mps_{base_name}_vrs'
    output_folder = os.path.join(folder_path, mps)
    return os.path.exists(output_folder) and os.path.isdir(output_folder)

def get_driving_feedback(results_dict, mirror_scores, overall_score):

    """ Generate driving feedback using the local LLM based on the results dictionary and mirror scores."""

    prompt = generate_driving_feedback_prompt(results_dict, mirror_scores, overall_score)

    full_prompt = system_prompt + "\n\n" + prompt + 'Ensure your response fits directly within the designated tokens.'
    
    with st.spinner("🧠 Analyzing driving performance..."):
        # For local LLM APIs (like LlamaCpp)
        response = llm.create_completion(
            prompt=full_prompt,
            max_tokens=600
        )
        
        feedback = response['choices'][0]['text'].strip()
        st.session_state.tips_output = feedback
        
    return feedback




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



# Process the data when the button is clicked
if process_btn and not st.session_state.processing:
    st.session_state.review_mode = False  
    st.session_state.preview_mode = False
    st.session_state.llm_mode = False

    vrs_file = find_vrs_file(folder_path)

    if vrs_file:
        st.info(f"🗂️ Processing file: **{vrs_file}**")

        base_name = os.path.splitext(vrs_file)[0]  # filename without extension

        input_dir = os.path.expanduser(folder_path)
        cmd_mps = ["aria_mps", "single", "-i", input_dir]
        full_vrs_path = os.path.join(input_dir, vrs_file)
        preds_path = f'{folder_path}/gaze_predictions.csv'
        cmd_gaze = ['python', 'MSc_AI_Thesis/projectaria_client_sdk_samples/projectaria_eyetracking/projectaria_eyetracking/model_inference_demo.py' , '--vrs' , full_vrs_path , 
                    '--output_file' , preds_path , '-c']
        

        if check_files_exist(files_to_check):
            st.success("✅ All required files already present!")
            st.session_state.vde = VRSDataExtractor(full_vrs_path)
            dp = DataProcessor(st.session_state.vde)
            st.session_state.results_dict = dp.vrs_processing(full_vrs_path, callbacks={
            "object_detection": object_detection_progress,
            "image_extraction": frame_progress_callback,
            "driving_evaluation": driver_evaluation_progress
            }, preds_path=None)       
        else:
            st.info("🔄 Retrieving Gaze Estimations")
            try:
                subprocess.Popen(cmd_gaze)  # non-blocking
                st.session_state.processing = True
                st.session_state.start_time = time.time()
                st.session_state.stop_waiting = False
                st.session_state.process_finished = False
                st.session_state.base_name = base_name  # save base name for output check
            except Exception as e:
                st.error(f"❌ Failed to run aria_mps: {e}")

            st.session_state.vde = VRSDataExtractor(full_vrs_path)
            dp = DataProcessor(st.session_state.vde)
            st.session_state.results_dict = dp.vrs_processing(full_vrs_path, callbacks={
            "object_detection": object_detection_progress,
            "image_extraction": frame_progress_callback,
            "driving_evaluation": driver_evaluation_progress
            }, preds_path=preds_path) 

        
    else:
        st.warning("⚠️ No .vrs files found in the selected folder.")

# Preview the annotated video data when the button is clicked
if preview_btn:
    st.session_state.review_mode = False 
    st.session_state.llm_mode = False
    st.session_state.preview_mode = True


    st.info("🔍 Previewing Video Data")
    vid = open(st.session_state.results_dict['video_path'], 'rb')
    video_bytes = vid.read()

    if st.session_state.results_dict is None or not st.session_state.results_dict:
        st.warning("⚠️ No file found to preview.")
    else:
        st.video(video_bytes)

# Show the session scores when the button is clicked
if score_btn:
    if not st.session_state.results_dict or 'scores' not in st.session_state.results_dict:
        st.warning("⚠️ No scores available. Please process the data first.")
    else:
        st.session_state.review_mode = False
        st.session_state.preview_mode = False
        st.session_state.llm_mode = False

        with st.spinner("🏅 Session Results"):
            time.sleep(0)  # Simulate processing time

        
        st.markdown(f"### 🧾 Score Summary for Session `{selected_folder}`")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Overall Score", f"{100 * st.session_state.results_dict['scores']['score']:.2f}%")
            st.metric("Left Mirror Checks", f"{100 * st.session_state.results_dict['scores']['lw_score']:.2f}%")
            st.metric("Right Mirror Checks", f"{100 * st.session_state.results_dict['scores']['rw_score']:.2f}%")
            st.metric("Rearview Mirror Checks", f"{100 * st.session_state.results_dict['scores']['rv_score']:.2f}%")
            st.metric("Mean Gaze Fixation", f"{st.session_state.results_dict['mean_gaze_fixation_duration_sec']:.2f} sec")

        with col2:
            st.metric("🚗 One-Handed Driving", f"{st.session_state.results_dict['one_handed_percent']:.2f}%")
            st.metric("🫱 Crossover Steering", f"{st.session_state.results_dict['crossover_occurrences']} times")
            st.metric("🎛️ Infotainment Distractions", f"{st.session_state.results_dict['infotainment_distraction_count']} times")
            st.metric("📷 Speedometer Checks", f"{st.session_state.results_dict['speedometer_checks']} times")
            st.metric("📱 Mobile Phone Usage", f"{st.session_state.results_dict['mobile_phone_usage_count']} times")

        st.markdown("---")

# Provide tips for improvement when the button is clicked
if tips_btn:

    if not st.session_state.results_dict or 'scores' not in st.session_state.results_dict:
        st.warning("⚠️ No scores available. Please process the data first.")
    else:
        st.session_state.processing = False
        st.session_state.review_mode = False
        st.session_state.preview_mode = False
        st.session_state.llm_mode = True

        overall_score = st.session_state.results_dict['scores']['score']
        left_mirror_score = st.session_state.results_dict['scores']['lw_score']
        right_mirror_score = st.session_state.results_dict['scores']['rw_score']
        rearview_mirror_score = st.session_state.results_dict['scores']['rv_score']
        # mobile_phone_count = st.session_state.results_dict['scores']

        system_prompt = """You are a professional driving instructor with 15+ years of experience. Your role is to analyze driving session data and provide targeted, 
                            actionable feedback.

                        INSTRUCTIONS:
                        - Only focus on areas scoring below 80 or showing concerning behavior
                        - Provide exactly 3 specific, practical tips per weak area identified
                        - Use clear, direct language that a student can immediately apply
                        - Prioritize safety-critical issues first
                        - Be encouraging but honest about areas needing improvement

                        RESPONSE FORMAT:
                        For each weak area, structure your response as:
                        **[Behavior Name]** (Score: X%)
                        1. [Specific tip with clear action]
                        2. [Specific tip with clear action] 
                        3. [Specific tip with clear action]

                        If all scores are above 90%, acknowledge good performance and suggest one advanced technique."""

            
        output = get_driving_feedback(st.session_state.results_dict, 
                                        {'left': left_mirror_score,
                                        'right': right_mirror_score,
                                        'rearview': rearview_mirror_score},
                                        overall_score)      
        
# review mistakes when the button is clicked      
if review_btn:
    if not st.session_state.results_dict or 'mistake_videos_directory' not in st.session_state.results_dict:
        st.warning("⚠️ No mistakes found in the session. Please process the data first.")
    else:
        # Set the mode BEFORE any UI elements
        st.session_state.preview_mode = False
        st.session_state.llm_mode = False
        st.session_state.review_mode = True

# Check if review mode is active and results data is available
if hasattr(st.session_state, 'review_mode') and st.session_state.review_mode:
    if st.session_state.results_dict and 'mistake_videos_directory' in st.session_state.results_dict:
        try:
            # Get video paths
            mistake_videos_dir = st.session_state.results_dict['mistake_videos_directory']
            
            if not os.path.exists(mistake_videos_dir):
                st.error("❌ Mistake videos directory not found.")
                st.session_state.review_mode = False
            else:
                # Get all video files
                all_files = os.listdir(mistake_videos_dir)
                mistake_video_paths = [
                    os.path.join(mistake_videos_dir, f) 
                    for f in all_files 
                    if f.endswith('.mp4') and not f.startswith('.')
                ]
                
                if not mistake_video_paths:
                    st.warning("⚠️ No mistake videos found in the directory.")
                    st.session_state.review_mode = False
                else:
                    # Create a more user-friendly display function
                    def format_video_name(path):
                        filename = os.path.basename(path)
                        # Remove file extension and replace underscores with spaces
                        name = filename.replace('.mp4', '').replace('_', ' ').title()
                        # Make it more readable
                        name = name.replace('Montage', 'Review').replace('Usage', 'Usage')
                        return name
                    
                    # Add header for the review section
                    st.markdown("## 🎥 Mistake Video Review")
                    st.markdown("Select a mistake category to review the detected incidents:")
                    
                    # Initialize selected video in session state if not exists
                    if 'selected_mistake_video' not in st.session_state:
                        st.session_state.selected_mistake_video = mistake_video_paths[0]
                    
                    # Create the selectbox with a unique key
                    selected_video = st.selectbox(
                        "Select a mistake category to review:",
                        options=mistake_video_paths,
                        format_func=format_video_name,
                        key="mistake_video_selector",
                        index=mistake_video_paths.index(st.session_state.selected_mistake_video) 
                        if st.session_state.selected_mistake_video in mistake_video_paths else 0
                    )
                    
                    # Update session state
                    st.session_state.selected_mistake_video = selected_video
                    
                    if selected_video:
                        try:
                            # Add spacing and styling
                            st.markdown("---")
                            
                            # Create columns for better layout
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.subheader(f"📹 {format_video_name(selected_video)}")
                            
                            with col2:
                                # Add file info
                                file_size = os.path.getsize(selected_video)
                                st.caption(f"File size: {file_size / (1024*1024):.1f} MB")
                            
                            # Check if file exists and is readable
                            if not os.path.exists(selected_video):
                                st.error("❌ Selected video file not found.")
                            elif not os.access(selected_video, os.R_OK):
                                st.error("❌ Cannot read the selected video file.")
                            else:
                                # Show loading message for large videos
                                if file_size > 10 * 1024 * 1024:  # 10MB
                                    with st.spinner("Loading video... (Large file detected)"):
                                        with open(selected_video, 'rb') as video_file:
                                            video_bytes = video_file.read()
                                else:
                                    with open(selected_video, 'rb') as video_file:
                                        video_bytes = video_file.read()
                                
                                # Create a stable container for the video
                                video_container = st.container()
                                
                                with video_container:
                                    st.video(video_bytes, format='video/mp4')
                                
                                # Add additional info below video
                                st.markdown("### 📊 Video Information")
                                video_info_col1, video_info_col2 = st.columns(2)
                                
                                with video_info_col1:
                                    mistake_type = os.path.basename(selected_video).replace('.mp4', '').replace('_montage', '')
                                    st.info(f"**Mistake Type:** {mistake_type.replace('_', ' ').title()}")
                                
                                with video_info_col2:
                                    st.info(f"**File:** {os.path.basename(selected_video)}")
                                
                                # Add action buttons
                                button_col1, button_col2, button_col3 = st.columns(3)
                                
                                with button_col1:
                                    if st.button("🔄 Reload Video", key="reload_video"):
                                        st.rerun()
                                
                                with button_col2:
                                    if st.button("📁 Open Folder", key="open_folder"):
                                        try:
                                            import subprocess
                                            import platform
                                            
                                            if platform.system() == "Windows":
                                                subprocess.run(['explorer', mistake_videos_dir])
                                            elif platform.system() == "Darwin":  # macOS
                                                subprocess.run(['open', mistake_videos_dir])
                                            else:  # Linux
                                                subprocess.run(['xdg-open', mistake_videos_dir])
                                            
                                            st.success("📁 Opened folder in file explorer")
                                        except Exception as e:
                                            st.error(f"Cannot open folder: {e}")
                                
                                with button_col3:
                                    if st.button("❌ Close Review", key="close_review"):
                                        st.session_state.review_mode = False
                                        st.rerun()
                        
                        except FileNotFoundError:
                            st.error(f"❌ Video file not found: {selected_video}")
                        except PermissionError:
                            st.error(f"❌ Permission denied accessing: {selected_video}")
                        except Exception as e:
                            st.error(f"❌ Error loading video: {str(e)}")
                            st.info("💡 Try refreshing the page or check if the video file is corrupted.")
        
        except Exception as e:
            st.error(f"❌ Error in video review setup: {str(e)}")
            st.session_state.review_mode = False
    else:
        st.warning("⚠️ No results data available for review.")
        st.session_state.review_mode = False



# Auto-refresh every 30 seconds (30000 milliseconds)
# if not st.session_state.processing and not st.session_state.review_mode and not st.session_state.preview_mode and not st.session_state.llm_mode:
#     st_autorefresh(interval=30000, key="datarefresh")


# Check if processing is ongoing and files are being generated
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

# Show success message if processing is finished
if st.session_state.process_finished:
    st.success("You can now score the driver.")

# Show the tips for improvement if available
if st.session_state.tips_output:
    st.info("💡 Tips for driving improvement:")
    st.write(f"**Tips for improvement:** {st.session_state.tips_output}")


# Add some bottom padding
st.markdown("<br><br>", unsafe_allow_html=True)
