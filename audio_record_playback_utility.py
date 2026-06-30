import pyaudio
import numpy as np
import threading
import cv2
from collections import deque
import time
import wave
import os
from scipy import signal
import sys
import tty
import termios
import select

def get_char_nonblocking():
    """Get a character from stdin without blocking, return None if no char available"""
    try:
        import select
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            return ch
    except:
        pass
    return None

def setup_terminal():
    """Set terminal to raw mode to capture input without echo"""
    try:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setraw(fd)
        return old_settings
    except:
        return None

def restore_terminal(old_settings):
    """Restore terminal to normal mode"""
    if old_settings:
        try:
            fd = sys.stdin.fileno()
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except:
            pass

def main():
    # Audio parameters
    CHUNK = 256  # Smaller buffer for LOW LATENCY
    FORMAT = pyaudio.paFloat32  # Audio format
    CHANNELS = 1  # Mono (1 channel)
    RATE = 44100  # Sample rate in Hz
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    while True:  # Main menu loop
        # List all available audio devices
        print("\n" + "="*60)
        print("Available audio devices:")
        print("="*60)
        device_count = p.get_device_count()
        output_devices = []
        input_devices = []
        
        for i in range(device_count):
            info = p.get_device_info_by_index(i)
            max_output_channels = info['maxOutputChannels']
            max_input_channels = info['maxInputChannels']
            
            if max_output_channels > 0:
                output_devices.append((i, info['name']))
                print(f"[{i}] OUTPUT: {info['name']} ({max_output_channels} channels)")
            
            if max_input_channels > 0:
                input_devices.append((i, info['name']))
                print(f"[{i}] INPUT:  {info['name']} ({max_input_channels} channels)")
        
        print("="*60)
        
        # Get user input for devices (only once per session)
        if 'output_device_index' not in locals():
            while True:
                try:
                    output_choice = input(f"\nEnter OUTPUT device number (0-{device_count-1}): ").strip()
                    output_device_index = int(output_choice)
                    if 0 <= output_device_index < device_count:
                        output_info = p.get_device_info_by_index(output_device_index)
                        if output_info['maxOutputChannels'] > 0:
                            print(f"✓ Selected OUTPUT: {output_info['name']}")
                            break
                    print("Invalid output device!")
                except ValueError:
                    print("Please enter a valid number!")
        
            while True:
                try:
                    input_choice = input(f"\nEnter INPUT device number (0-{device_count-1}): ").strip()
                    input_device_index = int(input_choice)
                    if 0 <= input_device_index < device_count:
                        input_info = p.get_device_info_by_index(input_device_index)
                        if input_info['maxInputChannels'] > 0:
                            print(f"✓ Selected INPUT: {input_info['name']}")
                            break
                    print("Invalid input device!")
                except ValueError:
                    print("Please enter a valid number!")
        
        print("\nStarting microphone input and loudness display...")
        print("Press 'q' to go back to mode selection.")
        
        # Store audio data for visualization
        loudness_history = deque(maxlen=50)  # Reduced history
        current_loudness = [0]  # Use list for thread-safe access
        is_running = [True]
        last_print_time = [time.time()]
        
        # Mode selection
        print("\n" + "="*60)
        print("Select mode:")
        print("[1] Passthrough (listen to microphone)")
        print("[2] Record audio")
        print("[3] Playback recorded audio")
        print("[4] Volume settings")
        print("[q] Exit program")
        print("="*60)
        
        # Volume control variable (shared across modes)
        if 'volume_gain_db' not in locals():
            volume_gain_db = [0]  # Use list for easier modification
        
        while True:
            mode_choice = input("Enter mode (1-4 or q): ").strip().lower()
            if mode_choice == 'q':
                print("\nExiting program...")
                p.terminate()
                return
            if mode_choice == '4':
                # Volume settings menu
                print("\n" + "="*60)
                print("Volume Settings")
                print("="*60)
                print(f"Current Volume: {volume_gain_db[0]:+.1f} dB")
                print("Use UP/DOWN arrow keys to adjust volume")
                print("Press ENTER to confirm")
                print("="*60)
                
                if sys.platform == 'darwin':  # macOS
                    fd = sys.stdin.fileno()
                    old_settings = termios.tcgetattr(fd)
                    try:
                        tty.setraw(fd)
                        while True:
                            ch = sys.stdin.read(1)
                            if ch == '\x1b':  # Escape sequence
                                next1 = sys.stdin.read(1)
                                next2 = sys.stdin.read(1)
                                if next1 == '[':
                                    if next2 == 'A':  # Up arrow
                                        volume_gain_db[0] = min(24.0, volume_gain_db[0] + 0.5)
                                        print(f"\rCurrent Volume: {volume_gain_db[0]:+.1f} dB", end='', flush=True)
                                    elif next2 == 'B':  # Down arrow
                                        volume_gain_db[0] = max(-24.0, volume_gain_db[0] - 0.5)
                                        print(f"\rCurrent Volume: {volume_gain_db[0]:+.1f} dB", end='', flush=True)
                            elif ch == '\r':  # Enter
                                print("\n✓ Volume saved!")
                                break
                    finally:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                        # Flush any remaining input
                        while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                            sys.stdin.read(1)
                continue
            
            if mode_choice in ['1', '2', '3']:
                mode = int(mode_choice)
                break
            print("Invalid choice!")
        
        record_filename = "audio_recording.wav"
        is_recording = mode == 2
        is_playback = mode == 3
        
        # If playback mode, let user select which file to play
        if is_playback:
            # Find all WAV files in current directory
            wav_files = [f for f in os.listdir('.') if f.endswith('.wav')]
            
            if not wav_files:
                print("No WAV files found in current directory!")
                continue
            
            print("\n" + "="*60)
            print("Available WAV files:")
            print("="*60)
            for i, filename in enumerate(wav_files, 1):
                print(f"[{i}] {filename}")
            print("="*60)
            
            while True:
                try:
                    choice = input(f"Select file to play (1-{len(wav_files)}): ").strip()
                    file_index = int(choice) - 1
                    if 0 <= file_index < len(wav_files):
                        record_filename = wav_files[file_index]
                        print(f"✓ Selected: {record_filename}")
                        break
                    print("Invalid selection!")
                except ValueError:
                    print("Please enter a valid number!")
        
        # Open input stream (microphone)
        stream_in = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=CHUNK
        )
        
        # Open output stream (speaker)
        stream_out = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
            output_device_index=output_device_index,
            frames_per_buffer=CHUNK
        )
        
        def apply_noise_reduction(audio_chunk):
            """Apply high-pass filter to reduce background noise"""
            # Convert to numpy array if needed
            audio_np = np.frombuffer(audio_chunk, dtype=np.float32)
            
            # Create high-pass filter to remove low frequency noise
            # Butterworth filter, 2nd order, cutoff at 300 Hz
            sos = signal.butter(2, 300, 'hp', fs=RATE, output='sos')
            filtered = signal.sosfilt(sos, audio_np)
            
            # Apply gentle noise gate - reduce very quiet sounds
            # Much higher threshold to avoid clipping
            threshold = 0.02
            mask = np.abs(filtered) > threshold
            filtered = filtered * mask
            
            # Apply soft compression to prevent clipping
            compressed = np.tanh(filtered * 0.8) / 0.8
            
            return compressed.astype(np.float32).tobytes()
        
        def apply_gain(audio_chunk, gain_db=6.0):
            """Apply gain boost to quiet audio - NO clipping, just linear scaling"""
            audio_np = np.frombuffer(audio_chunk, dtype=np.float32)
            gain_linear = 10 ** (gain_db / 20.0)
            boosted = audio_np * gain_linear
            
            # NO clipping at all - let it be natural
            # If it goes above 1.0, the system will handle it (speaker will limit)
            return boosted.astype(np.float32).tobytes()
        
        def audio_processing():
            """Thread function to process audio and calculate loudness"""
            recorded_frames = []
            
            if is_playback and os.path.exists(record_filename):
                # Playback mode
                with wave.open(record_filename, 'rb') as wav_file:
                    print(f"Playing: {record_filename}")
                    while is_running[0]:
                        frames = wav_file.readframes(CHUNK)
                        if not frames:
                            print("Playback finished!")
                            break
                        
                        if not is_running[0]:
                            break
                        
                        # NO base gain - use W/S keys to control volume
                        boosted_audio = apply_gain(frames, gain_db=volume_gain_db[0])
                        stream_out.write(boosted_audio)
                        
                        # Calculate loudness from original (before boost)
                        audio_np = np.frombuffer(frames, dtype=np.float32)
                        rms = np.sqrt(np.mean(audio_np**2))
                        db = 20 * np.log10(rms + 1e-10)
                        db_normalized = max(0, min(100, db + 80))
                        current_loudness[0] = db_normalized
                        loudness_history.append(db_normalized)
            elif not os.path.exists(record_filename) and is_playback:
                print(f"Error: {record_filename} not found!")
                return
            
            elif is_recording:
                # Record mode
                print(f"Recording to: {record_filename}")
                while is_running[0]:
                    try:
                        audio_data = stream_in.read(CHUNK, exception_on_overflow=False)
                        
                        if not is_running[0]:
                            break
                        
                        # Apply noise reduction
                        clean_audio = apply_noise_reduction(audio_data)
                        recorded_frames.append(clean_audio)
                        
                        audio_np = np.frombuffer(audio_data, dtype=np.float32)
                        rms = np.sqrt(np.mean(audio_np**2))
                        db = 20 * np.log10(rms + 1e-10)
                        db_normalized = max(0, min(100, db + 80))
                        current_loudness[0] = db_normalized
                        loudness_history.append(db_normalized)
                        
                    except Exception as e:
                        print(f"Recording error: {e}")
                        break
                
                # Save recording
                if recorded_frames:
                    with wave.open(record_filename, 'wb') as wav_file:
                        wav_file.setnchannels(CHANNELS)
                        wav_file.setsampwidth(p.get_sample_size(FORMAT))
                        wav_file.setframerate(RATE)
                        wav_file.writeframes(b''.join(recorded_frames))
                    
                    # Restore terminal to normal mode before asking for input
                    if old_terminal_settings:
                        try:
                            termios.tcsetattr(fd, termios.TCSADRAIN, old_terminal_settings)
                        except:
                            pass
                    
                    # Ask user for a custom filename
                    print(f"Recording saved temporarily as: {record_filename}")
                    custom_name = input("Enter a name for this recording (without .wav): ").strip()
                    if custom_name:
                        custom_filename = f"{custom_name}.wav"
                        try:
                            os.rename(record_filename, custom_filename)
                            print(f"✓ Recording saved as: {custom_filename}")
                        except Exception as e:
                            print(f"Could not rename file: {e}")
                            print(f"Recording saved as: {record_filename}")
                    else:
                        print(f"Recording saved as: {record_filename}")
                    
                    # Put terminal back in raw mode for the display loop
                    try:
                        tty.setraw(fd)
                    except:
                        pass
                return
            
            else:
                # Passthrough mode
                while is_running[0]:
                    try:
                        audio_data = stream_in.read(CHUNK, exception_on_overflow=False)
                        # NO base gain - use W/S keys to control volume
                        boosted_audio = apply_gain(audio_data, gain_db=volume_gain_db[0])
                        stream_out.write(boosted_audio)
                        
                        audio_np = np.frombuffer(audio_data, dtype=np.float32)
                        rms = np.sqrt(np.mean(audio_np**2))
                        db = 20 * np.log10(rms + 1e-10)
                        db_normalized = max(0, min(100, db + 80))
                        current_loudness[0] = db_normalized
                        loudness_history.append(db_normalized)
                        
                    except Exception as e:
                        print(f"Audio error: {e}")
                        break
        
        # Start audio processing thread
        audio_thread = threading.Thread(target=audio_processing, daemon=True)
        audio_thread.start()
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        # Set terminal to raw mode for arrow key capture
        fd = sys.stdin.fileno()
        old_terminal_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
        except:
            old_terminal_settings = None
        
        # Simple display loop
        try:
            mode_name = ["", "Passthrough", "Recording", "Playback"][mode]
            
            while is_running[0]:
                # Read camera frame
                ret, frame = cap.read()
                
                if ret:
                    # Calculate values EVERY FRAME (NO 0.5s DELAY!)
                    loudness = current_loudness[0]
                    bar_length = int(loudness / 5)  # 0-20 characters
                    bar = '*' * bar_length + ' ' * (20 - bar_length)
                    
                    # Calculate statistics
                    if loudness_history:
                        mean_loudness = np.mean(list(loudness_history))
                        max_loudness = np.max(list(loudness_history))
                        min_loudness = np.min(list(loudness_history))
                    else:
                        mean_loudness = max_loudness = min_loudness = 0
                    
                    # Determine level and color based on loudness
                    if loudness < 30:
                        level = "Quiet"
                        color = (0, 255, 0)  # Green
                    elif loudness < 60:
                        level = "Normal"
                        color = (0, 255, 255)  # Yellow
                    elif loudness < 80:
                        level = "Loud"
                        color = (0, 165, 255)  # Orange
                    else:
                        level = "VERY LOUD!"
                        color = (0, 0, 255)  # Red
                    
                    # Draw text EVERY frame
                    cv2.putText(frame, f"Mode: {mode_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(frame, f"Level: {level}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(frame, f"Current: {loudness:.1f} dB", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(frame, f"Mean: {mean_loudness:.1f} dB | Max: {max_loudness:.1f} dB | Min: {min_loudness:.1f} dB", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
                    cv2.putText(frame, f"[{bar}]", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(frame, f"Volume: {volume_gain_db[0]:+.1f} dB | W/S to adjust, Q to exit", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
                    
                    # Display frame
                    cv2.imshow('Camera Feed with Audio Monitor', frame)
                
                # Fast keyboard input via cv2
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\nReturning to menu...")
                    is_running[0] = False
                    cv2.destroyWindow('Camera Feed with Audio Monitor')
                    break
                elif key == ord('w') or key == ord('W'):  # W key - increase volume
                    volume_gain_db[0] = min(24.0, volume_gain_db[0] + 0.5)
                elif key == ord('s') or key == ord('S'):  # S key - decrease volume
                    volume_gain_db[0] = max(-24.0, volume_gain_db[0] - 0.5)
                
                # Check if audio thread has finished (for playback/recording completion)
                if not audio_thread.is_alive():
                    print("\nAudio operation completed.")
                    is_running[0] = False
                    break
                
        except KeyboardInterrupt:
            print("\n\nStopping audio stream...")
            is_running[0] = False
        except Exception as e:
            print(f"\nError: {e}")
            is_running[0] = False
        finally:
            # Restore terminal settings
            if old_terminal_settings:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_terminal_settings)
                except:
                    pass
            
            # Ensure we stop everything
            is_running[0] = False
            
            try:
                # Close all OpenCV windows first
                cv2.destroyAllWindows()
                for _ in range(5):
                    cv2.waitKey(1)  # Process events to actually close window
            except:
                pass
            
            try:
                # Cleanup camera
                if cap.isOpened():
                    cap.release()
            except:
                pass
            
            try:
                # Wait for audio thread to finish
                audio_thread.join(timeout=1)
            except:
                pass
            
            try:
                # Stop and close streams
                stream_in.stop_stream()
                stream_in.close()
            except:
                pass
            
            try:
                stream_out.stop_stream()
                stream_out.close()
            except:
                pass
            
            print("Audio stream and camera closed.")

if __name__ == "__main__":
    main()
