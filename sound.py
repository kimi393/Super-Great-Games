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
    CHUNK = 2048  # Optimized buffer size
    FORMAT = pyaudio.paFloat32  # Audio format
    CHANNELS = 1  # Mono (1 channel)
    RATE = 44100  # Sample rate in Hz
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
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
    
    # Get user input for devices
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
    print("Press 'q' to quit.")
    
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
    print("="*60)
    
    while True:
        mode_choice = input("Enter mode (1-3): ").strip()
        if mode_choice in ['1', '2', '3']:
            mode = int(mode_choice)
            break
        print("Invalid choice!")
    
    record_filename = "audio_recording.wav"
    is_recording = mode == 2
    is_playback = mode == 3
    
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
        threshold = 0.01
        mask = np.abs(filtered) > threshold
        filtered = filtered * mask
        
        return filtered.astype(np.float32).tobytes()
    
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
                    
                    stream_out.write(frames)
                    
                    # Calculate loudness from playback
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
                print(f"Recording saved to: {record_filename}")
            return
        
        else:
            # Passthrough mode
            while is_running[0]:
                try:
                    audio_data = stream_in.read(CHUNK, exception_on_overflow=False)
                    stream_out.write(audio_data)
                    
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
    
    # Simple display loop
    try:
        # Set up terminal for raw input (after all input() calls are done)
        old_terminal_settings = setup_terminal()
        
        mode_name = ["", "Passthrough", "Recording", "Playback"][mode]
        while is_running[0]:
            # Only print to console every 0.5 seconds to reduce I/O
            current_time = time.time()
            if current_time - last_print_time[0] > 1:
                loudness = current_loudness[0]
                bar_length = int(loudness / 5)  # 0-20 characters
                bar = '█' * bar_length + '░' * (20 - bar_length)
                
                # Determine level text
                if loudness < 30:
                    level = "Quiet"
                elif loudness < 60:
                    level = "Normal"
                elif loudness < 80:
                    level = "Loud"
                else:
                    level = "VERY LOUD!"
                
                status = "Press 'q' to quit"
                print(f"\r[{mode_name}] [{bar}] {loudness:5.1f} dB - {level:12s} | {status}", end='', flush=True)
                last_print_time[0] = current_time
            
            # Non-blocking keyboard check
            key = get_char_nonblocking()
            if key and key.lower() == 'q':
                print("\n\nExiting...")
                is_running[0] = False
                break
            
            # Check if audio thread has finished (for playback/recording completion)
            if not audio_thread.is_alive():
                print("\n\nAudio operation completed. Exiting...")
                is_running[0] = False
                break
            
            time.sleep(0.01)  # Small sleep to prevent CPU spinning
            
    except KeyboardInterrupt:
        print("\n\nStopping audio stream...")
    finally:
        # Restore terminal settings
        restore_terminal(old_terminal_settings)
        
        is_running[0] = False
        # Wait for audio thread to finish
        audio_thread.join(timeout=1)
        # Stop and close streams
        stream_in.stop_stream()
        stream_in.close()
        stream_out.stop_stream()
        stream_out.close()
        p.terminate()
        print("Audio stream closed.")

if __name__ == "__main__":
    main()
