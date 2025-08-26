import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
import pyttsx3
import threading
from model_handler import SignLanguageModel
import time
from datetime import datetime
from reference_guide import update_reference_guide

class HandSignLanguageGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("⚡ AI Gesture Recognition System ⚡")
        self.root.geometry("1400x900")
        self.root.configure(bg='#0a0a0a')  # Deep dark background
        self.root.resizable(True, True)
        
        # Configure styles
        self.setup_styles()
        
        # Initialize the sign language model
        self.model = SignLanguageModel()
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.last_spoken_text = ""
        self.speak_cooldown = 1.0
        self.last_speak_time = 0
        
        # Initialize voice toggle and recognition state
        self.voice_enabled = True
        self.is_recognizing = False
        
        # Initialize FPS calculation
        self.fps = 0
        self.fps_time = time.time()
        self.frame_count = 0

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        
        # Create GUI elements
        self.create_widgets()
        
        # Start video stream
        self.update_frame()
        
        self.root.mainloop()

    def setup_styles(self):
        # Configure modern futuristic styles for the GUI
        style = ttk.Style()
        
        # Main frame styles
        style.configure('Modern.TFrame', 
                       background='#0a0a0a',
                       relief='flat',
                       borderwidth=0)
        
        # Header styles
        style.configure('Header.TLabel', 
                       background='#0a0a0a', 
                       foreground='#00ffff',
                       font=('Orbitron', 20, 'bold'))
        
        style.configure('SubHeader.TLabel',
                       background='#0a0a0a',
                       foreground='#00ff88',
                       font=('Consolas', 16, 'bold'))
        
        # Status styles
        style.configure('Status.TLabel',
                       background='#0a0a0a',
                       foreground='#00ff00',
                       font=('Consolas', 14, 'bold'))
        
        style.configure('Info.TLabel',
                       background='#0a0a0a',
                       foreground='#88ffff',
                       font=('Consolas', 12))
        
        # Button styles
        style.configure('Modern.TButton',
                       background='#1a1a2e',
                       foreground='#ffffff',
                       font=('Consolas', 11, 'bold'),
                       borderwidth=2,
                       relief='raised',
                       padding=(15, 8))
        
        style.map('Modern.TButton',
                 background=[('active', '#16213e'),
                           ('pressed', '#0f3460')])
        
        style.configure('Active.TButton',
                       background='#00ff00',
                       foreground='#000000',
                       font=('Consolas', 11, 'bold'),
                       borderwidth=2,
                       relief='raised',
                       padding=(15, 8))
        
        style.map('Active.TButton',
                 background=[('active', '#00dd00'),
                           ('pressed', '#00bb00')])
        
        style.configure('Danger.TButton',
                       background='#ff3333',
                       foreground='#ffffff',
                       font=('Consolas', 11, 'bold'),
                       borderwidth=2,
                       relief='raised',
                       padding=(15, 8))
        
        # Panel styles
        style.configure('Panel.TFrame',
                       background='#111122',
                       relief='ridge',
                       borderwidth=2)

    def create_widgets(self):
        # Configure main grid weights for responsive design
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create main container with gradient-like effect
        self.main_frame = ttk.Frame(self.root, style='Modern.TFrame', padding="25")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.main_frame.grid_columnconfigure(0, weight=2)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        # Create title header with futuristic design
        self.header_frame = ttk.Frame(self.main_frame, style='Modern.TFrame')
        self.header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        self.title_label = ttk.Label(
            self.header_frame,
            text="⚡ AI GESTURE RECOGNITION SYSTEM ⚡",
            style='Header.TLabel'
        )
        self.title_label.pack(pady=10)

        # Create left panel for video and controls
        self.left_panel = ttk.Frame(self.main_frame, style='Panel.TFrame', padding="20")
        self.left_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 15))
        self.left_panel.grid_rowconfigure(1, weight=1)

        # Video section header
        self.video_header = ttk.Label(
            self.left_panel,
            text="📹 LIVE CAMERA FEED",
            style='SubHeader.TLabel'
        )
        self.video_header.grid(row=0, column=0, pady=(0, 15), sticky=tk.W)

        # Video display with enhanced border
        self.video_container = tk.Frame(self.left_panel, bg='#00ffff', bd=3, relief='ridge')
        self.video_container.grid(row=1, column=0, pady=(0, 20), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.video_label = tk.Label(self.video_container, bg='#000000', bd=2, relief='sunken')
        self.video_label.pack(padx=3, pady=3)

        # Status display section
        self.status_container = ttk.Frame(self.left_panel, style='Panel.TFrame', padding="15")
        self.status_container.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        self.status_container.grid_columnconfigure(1, weight=1)
        
        # Status header
        ttk.Label(self.status_container, text="📊 SYSTEM STATUS", style='SubHeader.TLabel').grid(
            row=0, column=0, columnspan=2, pady=(0, 10), sticky=tk.W)

        # Status indicators with better spacing
        ttk.Label(self.status_container, text="🎯 Detection:", style='Info.TLabel').grid(
            row=1, column=0, sticky=tk.W, padx=(0, 10))
        self.prediction_label = ttk.Label(
            self.status_container,
            text="⌛ Waiting for Gesture...",
            style='Status.TLabel'
        )
        self.prediction_label.grid(row=1, column=1, sticky=tk.W)

        ttk.Label(self.status_container, text="📈 Confidence:", style='Info.TLabel').grid(
            row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.confidence_label = ttk.Label(
            self.status_container,
            text="▫️ 0%",
            style='Status.TLabel'
        )
        self.confidence_label.grid(row=2, column=1, sticky=tk.W, pady=(5, 0))
        
        ttk.Label(self.status_container, text="⚡ FPS:", style='Info.TLabel').grid(
            row=3, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.fps_label = ttk.Label(
            self.status_container,
            text="0.0",
            style='Status.TLabel'
        )
        self.fps_label.grid(row=3, column=1, sticky=tk.W, pady=(5, 0))

        # Control panel with enhanced styling
        self.control_container = ttk.Frame(self.left_panel, style='Panel.TFrame', padding="15")
        self.control_container.grid(row=3, column=0, sticky=(tk.W, tk.E))
        self.control_container.grid_columnconfigure((0,1,2,3), weight=1)
        
        ttk.Label(self.control_container, text="🎮 CONTROLS", style='SubHeader.TLabel').grid(
            row=0, column=0, columnspan=4, pady=(0, 15), sticky=tk.W)

        self.start_button = ttk.Button(
            self.control_container,
            text="▶️ START",
            command=self.start_recognition,
            style='Active.TButton'
        )
        self.start_button.grid(row=1, column=0, padx=5, sticky=(tk.W, tk.E))

        self.stop_button = ttk.Button(
            self.control_container,
            text="⏹️ STOP",
            command=self.stop_recognition,
            style='Danger.TButton'
        )
        self.stop_button.grid(row=1, column=1, padx=5, sticky=(tk.W, tk.E))
        self.stop_button.state(['disabled'])

        self.clear_button = ttk.Button(
            self.control_container,
            text="🗑️ CLEAR",
            command=self.clear_history,
            style='Modern.TButton'
        )
        self.clear_button.grid(row=1, column=2, padx=5, sticky=(tk.W, tk.E))

        self.voice_button = ttk.Button(
            self.control_container,
            text="🔊 VOICE ON",
            command=self.toggle_voice,
            style='Active.TButton'
        )
        self.voice_button.grid(row=1, column=3, padx=5, sticky=(tk.W, tk.E))

        # Create right panel for history and gestures
        self.right_panel = ttk.Frame(self.main_frame, style='Panel.TFrame', padding="20")
        self.right_panel.grid(row=1, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        self.right_panel.grid_rowconfigure(1, weight=1)
        self.right_panel.grid_rowconfigure(3, weight=1)

        # History section
        ttk.Label(
            self.right_panel,
            text="📝 RECOGNITION HISTORY",
            style='SubHeader.TLabel'
        ).grid(row=0, column=0, pady=(0, 10), sticky=tk.W)
        
        self.history_container = tk.Frame(self.right_panel, bg='#00ffff', bd=2, relief='ridge')
        self.history_container.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
        
        self.history_text = scrolledtext.ScrolledText(
            self.history_container,
            width=40,
            height=15,
            font=('Consolas', 10),
            bg='#000011',
            fg='#00ff88',
            insertbackground='#00ff88',
            selectbackground='#003333',
            bd=0,
            wrap=tk.WORD
        )
        self.history_text.pack(padx=2, pady=2, fill=tk.BOTH, expand=True)

        # Gesture guide section
        ttk.Label(
            self.right_panel,
            text="📚 GESTURE GUIDE",
            style='SubHeader.TLabel'
        ).grid(row=2, column=0, pady=(0, 10), sticky=tk.W)
        
        self.reference_container = tk.Frame(self.right_panel, bg='#00ffff', bd=2, relief='ridge')
        self.reference_container.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.reference_text = scrolledtext.ScrolledText(
            self.reference_container,
            width=40,
            height=15,
            font=('Consolas', 9),
            bg='#000011',
            fg='#88ffff',
            insertbackground='#88ffff',
            selectbackground='#003333',
            bd=0,
            wrap=tk.WORD
        )
        self.reference_text.pack(padx=2, pady=2, fill=tk.BOTH, expand=True)
        self.populate_reference_guide()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            
            # Calculate FPS with smoothing
            current_time = time.time()
            if current_time - self.fps_time > 0:
                fps = 1 / (current_time - self.fps_time)
                self.fps = 0.9 * self.fps + 0.1 * fps  # Smooth FPS
                if self.frame_count % 10 == 0:  # Update FPS display every 10 frames
                    self.fps_label.config(text=f"{self.fps:.1f}")
            self.fps_time = current_time
            
            # Flip the frame horizontally for selfie view
            frame = cv2.flip(frame, 1)
            
            # Resize frame for better display
            frame = cv2.resize(frame, (640, 480))
            
            # Add futuristic border with gradient effect
            border_size = 15
            border_color = [0, 255, 255]  # Cyan border
            frame = cv2.copyMakeBorder(frame, border_size, border_size, border_size, border_size, 
                                     cv2.BORDER_CONSTANT, value=border_color)
            
            # Add corner decorations
            h, w = frame.shape[:2]
            # Top-left corner
            cv2.line(frame, (0, 30), (30, 30), (0, 255, 0), 3)
            cv2.line(frame, (30, 0), (30, 30), (0, 255, 0), 3)
            # Top-right corner
            cv2.line(frame, (w-30, 0), (w-30, 30), (0, 255, 0), 3)
            cv2.line(frame, (w-30, 30), (w, 30), (0, 255, 0), 3)
            # Bottom-left corner
            cv2.line(frame, (0, h-30), (30, h-30), (0, 255, 0), 3)
            cv2.line(frame, (30, h-30), (30, h), (0, 255, 0), 3)
            # Bottom-right corner
            cv2.line(frame, (w-30, h-30), (w, h-30), (0, 255, 0), 3)
            cv2.line(frame, (w-30, h-30), (w-30, h), (0, 255, 0), 3)
            
            # Convert BGR to RGB for MediaPipe processing
            rgb_frame = cv2.cvtColor(frame[border_size:-border_size, border_size:-border_size], cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe
            results = self.hands.process(rgb_frame)
            
            # Draw on the original frame (with border)
            display_frame = frame.copy()
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Adjust landmark coordinates for border offset
                    adjusted_landmarks = []
                    for landmark in hand_landmarks.landmark:
                        adjusted_landmarks.append([
                            landmark.x * (640) + border_size,
                            landmark.y * (480) + border_size,
                            landmark.z
                        ])
                    
                    # Create adjusted hand landmarks object
                    adjusted_hand = type('HandLandmarks', (), {})()
                    adjusted_hand.landmark = []
                    for x, y, z in adjusted_landmarks:
                        point = type('Point', (), {})()
                        point.x = x / w
                        point.y = y / h
                        point.z = z
                        adjusted_hand.landmark.append(point)
                    
                    # Draw enhanced hand landmarks
                    for i, landmark in enumerate(adjusted_landmarks):
                        x, y = int(landmark[0]), int(landmark[1])
                        # Draw landmarks with glow effect
                        cv2.circle(display_frame, (x, y), 8, (0, 255, 255), -1)
                        cv2.circle(display_frame, (x, y), 12, (0, 255, 255), 2)
                        cv2.circle(display_frame, (x, y), 16, (0, 150, 150), 1)
                    
                    # Draw connections
                    connections = self.mp_hands.HAND_CONNECTIONS
                    for connection in connections:
                        start_idx, end_idx = connection
                        start_pos = adjusted_landmarks[start_idx]
                        end_pos = adjusted_landmarks[end_idx]
                        cv2.line(display_frame, 
                               (int(start_pos[0]), int(start_pos[1])),
                               (int(end_pos[0]), int(end_pos[1])),
                               (255, 255, 255), 3)
                        cv2.line(display_frame, 
                               (int(start_pos[0]), int(start_pos[1])),
                               (int(end_pos[0]), int(end_pos[1])),
                               (0, 255, 0), 2)
                    
                    # Perform sign recognition
                    if self.is_recognizing:
                        predicted_sign, confidence = self.model.predict(hand_landmarks)
                        if predicted_sign:
                            prediction_text = f"🎯 {predicted_sign}"
                            confidence_text = f"📈 {confidence*100:.1f}%"
                            
                            self.prediction_label.config(text=prediction_text)
                            self.confidence_label.config(text=confidence_text)
                            
                            # Add to history
                            self.add_to_history(predicted_sign, confidence)
                            
                            # Speak the prediction if voice is enabled and it's a new gesture
                            self.speak_prediction(predicted_sign)
                            
                            # Display prediction on video
                            cv2.putText(display_frame, f"{predicted_sign} ({confidence*100:.1f}%)", 
                                      (border_size + 10, border_size + 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        else:
                            self.prediction_label.config(text="❓ Uncertain")
                            self.confidence_label.config(text="📉 Low")
            else:
                self.prediction_label.config(text="⌛ Waiting...")
                self.confidence_label.config(text="▫️ 0%")

            # Convert frame to PhotoImage and display
            img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        # Schedule the next update
        self.video_label.after(10, self.update_frame)

    def start_recognition(self):
        self.is_recognizing = True
        self.start_button.state(['disabled'])
        self.stop_button.state(['!disabled'])
        self.prediction_label.config(text="🎯 Ready to Detect")
        self.speak_prediction("Recognition started")
        
        # Update button styles
        self.start_button.configure(style='Modern.TButton')
        self.stop_button.configure(style='Active.TButton')

    def stop_recognition(self):
        self.is_recognizing = False
        self.start_button.state(['!disabled'])
        self.stop_button.state(['disabled'])
        self.prediction_label.config(text="⏸️ Recognition Paused")
        self.speak_prediction("Recognition stopped")
        
        # Update button styles
        self.stop_button.configure(style='Modern.TButton')
        self.start_button.configure(style='Active.TButton')

    def clear_history(self):
        self.history_text.delete(1.0, tk.END)
        self.history_text.insert(tk.END, "🧹 History cleared\n" + "─"*30 + "\n")

    def add_to_history(self, sign, confidence):
        # Add timestamp and prediction to history with modern formatting
        timestamp = datetime.now().strftime("%H:%M:%S")
        confidence_bar = "█" * int(confidence * 10) + "▒" * (10 - int(confidence * 10))
        history_entry = f"⏱️ [{timestamp}]\n🔍 {sign}\n📊 {confidence_bar} {confidence*100:.1f}%\n{'─'*30}\n"
        self.history_text.insert(tk.END, history_entry)
        self.history_text.see(tk.END)  # Scroll to the latest entry

    def populate_reference_guide(self):
        # Add comprehensive gesture reference guide with futuristic formatting
        reference_text = update_reference_guide()
        self.reference_text.insert(tk.END, reference_text)
        self.reference_text.config(state='disabled')  # Make it read-only
        reference_text = """📱 Supported Gestures 📱

🤚 Basic Gestures:
━━━━━━━━━━━━━━━━━━━━
👋 Open Palm      → Hello
✊ Closed Fist    → No
👍 Thumbs Up      → Good
👎 Thumbs Down    → Bad
👌 Pinched OK     → OK
✋ Spread Fingers → Stop
🤘 Rock Sign      → Rock

✨ Direction Gestures:
━━━━━━━━━━━━━━━━━━━━
👆 Point Up       → Up
👇 Point Down     → Down
👉 Point Right    → Right
👈 Point Left     → Left

� Number Gestures:
━━━━━━━━━━━━━━━━━━━━
☝️ One Finger     → 1
✌️ Two Fingers    → 2
👆👆👆 Three      → 3
�✌️ Four         → 4
🖐️ Five          → 5

� Special Gestures:
━━━━━━━━━━━━━━━━━━━━
🤙 Pinky & Thumb  → Call

🔤 ASL Alphabet:
━━━━━━━━━━━━━━━━━━━━
A → Closed fist, thumb aside
B → Open palm, fingers together
C → Curved open hand
D → 'D' shape with fingers
E → Fingers curled in
F → 'F' shape with fingers

💡 Pro Tips:
━━━━━━━━━━━━━━━━━━━━
✨ Keep hand steady & visible
🔦 Ensure good lighting
🎯 Center hand in camera
📊 Wait for 70%+ confidence
🎓 Make clear gestures
"""
        self.reference_text.insert(tk.END, reference_text)
        self.reference_text.config(state='disabled')  # Make it read-only

    def speak_prediction(self, text):
        """Speak the prediction if it's different from the last one and after cooldown"""
        import time
        current_time = time.time()
        
        if (self.voice_enabled and 
            text != self.last_spoken_text and 
            current_time - self.last_speak_time >= self.speak_cooldown):
            
            self.last_spoken_text = text
            self.last_speak_time = current_time
            
            # Run text-to-speech in a separate thread to avoid blocking
            threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()

    def _speak_thread(self, text):
        """Thread function for text-to-speech"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except:
            pass  # Ignore any speech errors

    def start_recognition(self):
        self.is_recognizing = True
        self.start_button.state(['disabled'])
        self.stop_button.state(['!disabled'])
        self.prediction_label.config(text="🎯 Ready to Detect")
        self.speak_prediction("Recognition started")
        
        # Update button styles
        self.start_button.configure(style='Modern.TButton')
        self.stop_button.configure(style='Danger.TButton')

    def stop_recognition(self):
        self.is_recognizing = False
        self.start_button.state(['!disabled'])
        self.stop_button.state(['disabled'])
        self.prediction_label.config(text="⏸️ Recognition Paused")
        self.speak_prediction("Recognition stopped")
        
        # Update button styles
        self.stop_button.configure(style='Modern.TButton')
        self.start_button.configure(style='Active.TButton')

    def clear_history(self):
        self.history_text.delete(1.0, tk.END)
        self.history_text.insert(tk.END, """╔══════════════════════════════════════╗
║         🧹 HISTORY CLEARED 🧹          ║
╚══════════════════════════════════════╝

📝 Recognition history will appear here...

""")

    def add_to_history(self, sign, confidence):
        # Add timestamp and prediction to history with futuristic formatting
        timestamp = datetime.now().strftime("%H:%M:%S")
        confidence_bar = "█" * int(confidence * 10) + "▒" * (10 - int(confidence * 10))
        
        # Color code based on confidence
        if confidence > 0.9:
            status = "🟢 EXCELLENT"
        elif confidence > 0.8:
            status = "🟡 GOOD"
        elif confidence > 0.7:
            status = "🟠 FAIR"
        else:
            status = "🔴 LOW"
            
        history_entry = f"""┌─────────────────────────────────────┐
│ ⏰ TIME: {timestamp}               │
│ 🎯 GESTURE: {sign:<20}       │
│ 📊 CONFIDENCE: {confidence*100:.1f}%           │
│ {confidence_bar} {status}      │
└─────────────────────────────────────┘

"""
        self.history_text.insert(tk.END, history_entry)
        self.history_text.see(tk.END)  # Scroll to the latest entry

    def toggle_voice(self):
        """Toggle voice output on/off with enhanced feedback"""
        self.voice_enabled = not self.voice_enabled
        icon = "🔊" if self.voice_enabled else "🔇"
        status = "ON" if self.voice_enabled else "OFF"
        self.voice_button.config(
            text=f"{icon} VOICE {status}",
            style='Active.TButton' if self.voice_enabled else 'Modern.TButton'
        )
        
        # Provide audio feedback
        if self.voice_enabled:
            self.speak_prediction("Voice output enabled")

    def __del__(self):
        self.cap.release()
        try:
            self.engine.stop()
        except:
            pass

if __name__ == "__main__":
    app = HandSignLanguageGUI()
