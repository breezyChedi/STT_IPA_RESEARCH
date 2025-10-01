#!/usr/bin/env python3
"""
Enhanced Audio Analysis Player for Wav2Seg Prediction Analysis
==============================================================

This script allows you to:
1. Loop through all prediction samples (worst/best)
2. Play audio files with boundary visualization
3. Display true vs predicted timestamps
4. Real-time playback cursor synchronized with audio
5. Interactive controls for navigation

Usage:
    python audio_analysis_player.py [--group worst|best|all] [--auto-play]

Controls:
    SPACE: Play/Pause
    N: Next sample
    P: Previous sample
    R: Replay current sample
    Q: Quit
    S: Save current plot
    T: Toggle timestamp display
    C: Toggle playback cursor
"""

import os
import sys
import json
import glob
import argparse
import threading
import time
from pathlib import Path

import numpy as np
import matplotlib
# Set backend before importing pyplot
matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import soundfile as sf
import sounddevice as sd
from datetime import datetime

class AudioAnalysisPlayer:
    def __init__(self, prediction_dir='prediction_analysis', group='all', auto_play=False, smooth_cursor=True):
        """
        Initialize the audio analysis player.
        
        Args:
            prediction_dir: Directory containing prediction analysis
            group: 'worst', 'best', or 'all'
            auto_play: Whether to auto-play audio when loading samples
            smooth_cursor: If True, use 60 FPS cursor updates; if False, use 30 FPS
        """
        self.prediction_dir = prediction_dir
        self.group = group
        self.auto_play = auto_play
        self.smooth_cursor = smooth_cursor
        self.current_index = 0
        self.samples = []
        self.current_audio = None
        self.current_sr = None
        self.is_playing = False
        self.playback_thread = None
        self.show_timestamps = True
        self.show_cursor = True
        
        # Playback cursor variables
        self.playback_start_time = None
        self.playback_position = 0.0
        self.cursor_line = None
        self.cursor_line2 = None
        self.audio_duration = 0.0
        self.cursor_update_thread = None
        self.stop_cursor_update = False
        
        # Cursor update settings
        if self.smooth_cursor:
            self.cursor_fps = 60  # Smooth 60 FPS
            self.cursor_sleep = 0.016  # ~16ms
        else:
            self.cursor_fps = 30  # Standard 30 FPS
            self.cursor_sleep = 0.033  # ~33ms
        
        # Load all samples
        self._load_samples()
        
        # Setup matplotlib
        plt.style.use('default')
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        
    def _load_samples(self):
        """Load all sample directories based on group selection."""
        print(f"🔍 Loading samples from '{self.prediction_dir}'...")
        
        if self.group == 'all':
            search_dirs = [
                os.path.join(self.prediction_dir, 'worst_predictions'),
                os.path.join(self.prediction_dir, 'best_predictions')
            ]
        elif self.group == 'worst':
            search_dirs = [os.path.join(self.prediction_dir, 'worst_predictions')]
        elif self.group == 'best':
            search_dirs = [os.path.join(self.prediction_dir, 'best_predictions')]
        else:
            raise ValueError("Group must be 'worst', 'best', or 'all'")
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                # Find all sample directories
                sample_dirs = glob.glob(os.path.join(search_dir, '*_*'))
                sample_dirs.sort()
                
                for sample_dir in sample_dirs:
                    if os.path.isdir(sample_dir):
                        # Find the audio file and metadata
                        wav_files = glob.glob(os.path.join(sample_dir, '*.wav'))
                        json_files = glob.glob(os.path.join(sample_dir, '*_metadata.json'))
                        
                        if wav_files and json_files:
                            sample_info = {
                                'dir': sample_dir,
                                'wav_file': wav_files[0],
                                'metadata_file': json_files[0],
                                'group': 'worst' if 'worst_predictions' in sample_dir else 'best',
                                'rank': int(os.path.basename(sample_dir).split('_')[0])
                            }
                            self.samples.append(sample_info)
        
        print(f"✅ Loaded {len(self.samples)} samples")
        if not self.samples:
            print("❌ No samples found! Make sure prediction_analysis directory exists.")
            sys.exit(1)
    
    def _load_current_sample(self):
        """Load the current sample's audio and metadata."""
        if not self.samples:
            return None, None, None
        
        sample = self.samples[self.current_index]
        
        # Load audio
        try:
            audio, sr = sf.read(sample['wav_file'])
            print(f"🎵 Loaded audio: {os.path.basename(sample['wav_file'])}")
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            return None, None, None
        
        # Load metadata
        try:
            with open(sample['metadata_file'], 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"📋 Loaded metadata for: {metadata['file_id']}")
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
            return None, None, None
        
        return audio, sr, metadata
    
    def _play_audio(self, audio, sr):
        """Play audio in a separate thread with timing tracking."""
        def play():
            try:
                print("🎯 Starting audio playback...")
                self.is_playing = True
                self.playback_start_time = time.time()
                self.playback_position = 0.0
                
                # Start the cursor animation
                if self.show_cursor:
                    print("🎯 Starting cursor updates...")
                    self._start_cursor_updates()
                
                # Play audio
                sd.play(audio, sr)
                print(f"🎵 Audio playing for {len(audio)/sr:.2f} seconds...")
                sd.wait()  # Wait until audio is finished
                
                print("🎵 Audio playback finished")
                self.is_playing = False
                self.playback_start_time = None
                
                # Stop cursor updates
                self._stop_cursor_updates()
                
            except Exception as e:
                print(f"❌ Audio playback error: {e}")
                self.is_playing = False
                self.playback_start_time = None
                self._stop_cursor_updates()
        
        # Stop any existing playback
        if self.playback_thread and self.playback_thread.is_alive():
            sd.stop()
            self.playback_thread.join()
        
        self.playback_thread = threading.Thread(target=play, daemon=True)
        self.playback_thread.start()
    
    def _stop_audio(self):
        """Stop audio playback and cursor animation."""
        print("⏸️ Stopping audio and cursor...")
        sd.stop()
        self.is_playing = False
        self.playback_start_time = None
        
        self._stop_cursor_updates()
        
        # Reset cursor position
        if self.cursor_line:
            try:
                self.cursor_line.set_xdata([0, 0])
            except:
                pass
        if self.cursor_line2:
            try:
                self.cursor_line2.set_xdata([0, 0])
            except:
                pass
        
        # Safe redraw
        try:
            if self.fig and hasattr(self.fig, 'canvas'):
                self.fig.canvas.draw_idle()
        except:
            pass
    
    def _start_cursor_updates(self):
        """Start the cursor update thread."""
        if self.show_cursor:
            print("🎯 Starting cursor update thread...")
            self.stop_cursor_update = False
            self.cursor_update_thread = threading.Thread(target=self._cursor_update_loop, daemon=True)
            self.cursor_update_thread.start()
    
    def _stop_cursor_updates(self):
        """Stop the cursor update thread."""
        print("🛑 Stopping cursor updates...")
        self.stop_cursor_update = True
        if self.cursor_update_thread and self.cursor_update_thread.is_alive():
            self.cursor_update_thread.join(timeout=1.0)
    
    def _cursor_update_loop(self):
        """Main cursor update loop running in separate thread."""
        print(f"🎯 Cursor update loop started ({self.cursor_fps} FPS)")
        frame_count = 0
        
        while not self.stop_cursor_update and self.is_playing:
            try:
                if self.playback_start_time:
                    # Calculate current playback position
                    elapsed_time = time.time() - self.playback_start_time
                    self.playback_position = min(elapsed_time, self.audio_duration)
                    
                    # Debug output every 1 second
                    if frame_count % self.cursor_fps == 0:  # Every FPS iterations (1 second)
                        print(f"🎯 Cursor position: {self.playback_position:.2f}s / {self.audio_duration:.2f}s")
                    
                    # Update cursor lines
                    if self.cursor_line:
                        self.cursor_line.set_xdata([self.playback_position, self.playback_position])
                        
                    if self.cursor_line2:
                        self.cursor_line2.set_xdata([self.playback_position, self.playback_position])
                    
                    # Force redraw on main thread
                    if self.fig and hasattr(self.fig, 'canvas'):
                        try:
                            # Schedule redraw on main thread
                            self.fig.canvas.draw_idle()
                        except:
                            pass
                
                frame_count += 1
                time.sleep(self.cursor_sleep)  # Update every ~16ms for 60 FPS smooth movement
                
            except Exception as e:
                print(f"⚠️ Cursor update error: {e}")
                break
        
        print("🎯 Cursor update loop finished")
    
    def _create_visualization(self, audio, sr, metadata):
        """Create the audio visualization with boundaries and playback cursor."""
        try:
            if self.fig is None:
                print("🎨 Creating new figure...")
                self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(16, 10))
                self.fig.suptitle('Audio Analysis Player with Real-time Cursor', fontsize=16, fontweight='bold')
                
                # Add control buttons
                self._add_control_buttons()
            
            # Clear previous plots
            self.ax1.clear()
            self.ax2.clear()
            
            # Store audio duration
            self.audio_duration = len(audio) / sr
            print(f"🎵 Audio duration: {self.audio_duration:.2f} seconds")
            
            # Get boundaries
            true_bounds = np.array(metadata['boundaries']['true_boundaries_seconds'])
            pred_bounds = np.array(metadata['boundaries']['predicted_boundaries_seconds'])
            
            # Plot waveform
            time = np.linspace(0, len(audio)/sr, len(audio))
            self.ax1.plot(time, audio, alpha=0.7, color='blue', linewidth=0.5)
            
            # Plot boundaries
            for bound in true_bounds:
                self.ax1.axvline(bound, color='limegreen', linestyle='-', alpha=0.9, linewidth=3, label='True Boundary' if bound == true_bounds[0] else "")
            for bound in pred_bounds:
                self.ax1.axvline(bound, color='crimson', linestyle='--', alpha=0.9, linewidth=3, label='Predicted Boundary' if bound == pred_bounds[0] else "")
            
            # Add playback cursor to waveform plot
            if self.show_cursor:
                print("🎯 Adding cursor lines to plots...")
                self.cursor_line = self.ax1.axvline(0, color='orange', linestyle='-', linewidth=3, alpha=0.9, label='Playback Position')
            else:
                self.cursor_line = None
            
            # Customize waveform plot
            sample_info = self.samples[self.current_index]
            title = f"[{sample_info['group'].upper()}] {metadata['file_id']} (F1: {metadata['evaluation_metrics']['f1']:.3f})"
            title += f" - Sample {self.current_index + 1}/{len(self.samples)}"
            self.ax1.set_title(title, fontsize=14, fontweight='bold')
            self.ax1.set_xlabel('Time (seconds)')
            self.ax1.set_ylabel('Amplitude')
            self.ax1.grid(True, alpha=0.3)
            
            # Create legend
            self.ax1.legend(loc='upper right', fontsize=10)
            
            # Plot boundary comparison
            y_true = np.ones(len(true_bounds))
            y_pred = np.ones(len(pred_bounds)) * 0.5
            
            self.ax2.scatter(true_bounds, y_true, color='limegreen', s=120, marker='|', 
                            label=f'True ({len(true_bounds)})', alpha=0.9, linewidth=3)
            self.ax2.scatter(pred_bounds, y_pred, color='crimson', s=120, marker='|', 
                            label=f'Predicted ({len(pred_bounds)})', alpha=0.9, linewidth=3)
            
            # Add playback cursor to boundary plot
            if self.show_cursor:
                self.cursor_line2 = self.ax2.axvline(0, color='orange', linestyle='-', linewidth=3, alpha=0.9)
            else:
                self.cursor_line2 = None
            
            self.ax2.set_ylim(0, 1.5)
            self.ax2.set_xlabel('Time (seconds)')
            self.ax2.set_ylabel('Boundary Type')
            self.ax2.set_title('Boundary Comparison with Playback Position')
            self.ax2.legend()
            self.ax2.grid(True, alpha=0.3)
            
            # Add metrics text
            cursor_status = "ON" if self.show_cursor else "OFF"
            metrics_text = f"""Metrics:
MAE: {metadata['evaluation_metrics']['mae']:.2f} frames
Precision: {metadata['evaluation_metrics']['precision']:.3f}
Recall: {metadata['evaluation_metrics']['recall']:.3f}
F1 Score: {metadata['evaluation_metrics']['f1']:.3f}

Audio Info:
Duration: {metadata['audio_info']['length_seconds']:.2f}s
Sample Rate: {metadata['audio_info']['sample_rate']} Hz
Speaker: {metadata['speaker_id']}

Playback Cursor: {cursor_status}

Transcription: {metadata['transcription'][:80]}{'...' if len(metadata['transcription']) > 80 else ''}"""
            
            self.ax2.text(1.02, 0.5, metrics_text, transform=self.ax2.transAxes, 
                         fontsize=10, verticalalignment='center',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
            plt.tight_layout()
            plt.subplots_adjust(right=0.75)  # Make room for text
            
            # Safe draw
            try:
                self.fig.canvas.draw()
                print("✅ Figure drawn successfully")
            except Exception as e:
                print(f"⚠️ Figure draw warning: {e}")
            
            # Display timestamps if enabled
            if self.show_timestamps:
                self._display_timestamps(true_bounds, pred_bounds)
                
        except Exception as e:
            print(f"❌ Visualization error: {e}")
            import traceback
            traceback.print_exc()
    
    def _display_timestamps(self, true_bounds, pred_bounds):
        """Display timestamps in console."""
        print("\n" + "="*80)
        print(f"📊 TIMESTAMPS - Sample {self.current_index + 1}/{len(self.samples)}")
        print("="*80)
        print("🎨 COLORS: True boundaries = LIME GREEN solid lines | Predicted boundaries = CRIMSON dashed lines")
        print("="*80)
        
        print(f"🟢 TRUE BOUNDARIES ({len(true_bounds)}) - LIME GREEN SOLID:")
        for i, bound in enumerate(true_bounds):
            print(f"   {i+1:2d}: {bound:8.3f}s", end="")
            if (i + 1) % 6 == 0:  # New line every 6 timestamps
                print()
        if len(true_bounds) % 6 != 0:
            print()
        
        print(f"\n🔴 PREDICTED BOUNDARIES ({len(pred_bounds)}) - CRIMSON DASHED:")
        for i, bound in enumerate(pred_bounds):
            print(f"   {i+1:2d}: {bound:8.3f}s", end="")
            if (i + 1) % 6 == 0:
                print()
        if len(pred_bounds) % 6 != 0:
            print()
        
        print("="*80)
    
    def _add_control_buttons(self):
        """Add interactive control buttons."""
        try:
            # Button positions
            button_height = 0.04
            button_width = 0.08
            button_y = 0.02
            
            # Create button axes
            ax_prev = plt.axes([0.05, button_y, button_width, button_height])
            ax_play = plt.axes([0.15, button_y, button_width, button_height])
            ax_next = plt.axes([0.25, button_y, button_width, button_height])
            ax_replay = plt.axes([0.35, button_y, button_width, button_height])
            ax_save = plt.axes([0.45, button_y, button_width, button_height])
            ax_timestamps = plt.axes([0.55, button_y, button_width, button_height])
            ax_cursor = plt.axes([0.65, button_y, button_width, button_height])
            ax_quit = plt.axes([0.75, button_y, button_width, button_height])
            
            # Create buttons
            self.btn_prev = Button(ax_prev, 'Previous')
            self.btn_play = Button(ax_play, 'Play/Pause')
            self.btn_next = Button(ax_next, 'Next')
            self.btn_replay = Button(ax_replay, 'Replay')
            self.btn_save = Button(ax_save, 'Save Plot')
            self.btn_timestamps = Button(ax_timestamps, 'Toggle TS')
            self.btn_cursor = Button(ax_cursor, 'Toggle Cursor')
            self.btn_quit = Button(ax_quit, 'Quit')
            
            # Connect button events
            self.btn_prev.on_clicked(lambda x: self.previous_sample())
            self.btn_play.on_clicked(lambda x: self.toggle_playback())
            self.btn_next.on_clicked(lambda x: self.next_sample())
            self.btn_replay.on_clicked(lambda x: self.replay_sample())
            self.btn_save.on_clicked(lambda x: self.save_plot())
            self.btn_timestamps.on_clicked(lambda x: self.toggle_timestamps())
            self.btn_cursor.on_clicked(lambda x: self.toggle_cursor())
            self.btn_quit.on_clicked(lambda x: self.quit_application())
            
            print("✅ Control buttons added successfully")
            
        except Exception as e:
            print(f"⚠️ Button creation warning: {e}")
    
    def next_sample(self):
        """Go to next sample."""
        if self.current_index < len(self.samples) - 1:
            self._stop_audio()
            self.current_index += 1
            self.display_current_sample()
        else:
            print("📍 Already at last sample")
    
    def previous_sample(self):
        """Go to previous sample."""
        if self.current_index > 0:
            self._stop_audio()
            self.current_index -= 1
            self.display_current_sample()
        else:
            print("📍 Already at first sample")
    
    def toggle_playback(self):
        """Toggle audio playback."""
        if self.is_playing:
            self._stop_audio()
            print("⏸️  Audio paused")
        else:
            if self.current_audio is not None:
                self._play_audio(self.current_audio, self.current_sr)
                print("▶️  Audio playing with real-time cursor")
    
    def replay_sample(self):
        """Replay current sample."""
        if self.current_audio is not None:
            self._stop_audio()
            time.sleep(0.1)  # Small delay to ensure cleanup
            self._play_audio(self.current_audio, self.current_sr)
            print("🔄 Replaying audio with cursor")
    
    def save_plot(self):
        """Save current plot."""
        if self.fig is not None:
            try:
                sample = self.samples[self.current_index]
                filename = f"analysis_{sample['group']}_{os.path.basename(sample['wav_file'])[:-4]}.png"
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"💾 Plot saved as: {filename}")
            except Exception as e:
                print(f"❌ Save error: {e}")
    
    def toggle_timestamps(self):
        """Toggle timestamp display."""
        self.show_timestamps = not self.show_timestamps
        status = "ON" if self.show_timestamps else "OFF"
        print(f"📊 Timestamp display: {status}")
        
        if self.show_timestamps and self.current_audio is not None:
            # Re-display timestamps for current sample
            audio, sr, metadata = self._load_current_sample()
            if metadata:
                true_bounds = np.array(metadata['boundaries']['true_boundaries_seconds'])
                pred_bounds = np.array(metadata['boundaries']['predicted_boundaries_seconds'])
                self._display_timestamps(true_bounds, pred_bounds)
    
    def toggle_cursor(self):
        """Toggle playback cursor display."""
        self.show_cursor = not self.show_cursor
        status = "ON" if self.show_cursor else "OFF"
        print(f"🎯 Playback cursor: {status}")
        
        # Refresh the current display
        if self.current_audio is not None:
            audio, sr, metadata = self._load_current_sample()
            if metadata:
                self._create_visualization(audio, sr, metadata)
    
    def quit_application(self):
        """Quit the application."""
        try:
            self._stop_audio()
            plt.close('all')
            print("👋 Goodbye!")
            sys.exit(0)
        except:
            sys.exit(0)
    
    def display_current_sample(self):
        """Display the current sample."""
        print(f"\n🎯 Loading sample {self.current_index + 1}/{len(self.samples)}...")
        
        # Load sample data
        audio, sr, metadata = self._load_current_sample()
        if audio is None:
            print("❌ Failed to load sample")
            return
        
        self.current_audio = audio
        self.current_sr = sr
        
        # Create visualization
        self._create_visualization(audio, sr, metadata)
        
        # Auto-play if enabled
        if self.auto_play:
            self._play_audio(audio, sr)
    
    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts."""
        def on_key(event):
            try:
                if event.key == ' ':  # Space
                    self.toggle_playback()
                elif event.key == 'n':  # Next
                    self.next_sample()
                elif event.key == 'p':  # Previous
                    self.previous_sample()
                elif event.key == 'r':  # Replay
                    self.replay_sample()
                elif event.key == 's':  # Save
                    self.save_plot()
                elif event.key == 't':  # Toggle timestamps
                    self.toggle_timestamps()
                elif event.key == 'c':  # Toggle cursor
                    self.toggle_cursor()
                elif event.key == 'q':  # Quit
                    self.quit_application()
            except Exception as e:
                print(f"⚠️ Keyboard event error: {e}")
        
        try:
            if self.fig:
                self.fig.canvas.mpl_connect('key_press_event', on_key)
                print("✅ Keyboard shortcuts connected")
        except Exception as e:
            print(f"⚠️ Keyboard setup warning: {e}")
    
    def run(self):
        """Run the audio analysis player."""
        print("🎵 Starting Enhanced Audio Analysis Player...")
        print("="*60)
        print("Controls:")
        print("  SPACE: Play/Pause    N: Next sample")
        print("  P: Previous sample   R: Replay current")
        print("  S: Save plot         T: Toggle timestamps")
        print("  C: Toggle cursor     Q: Quit")
        print("="*60)
        print("🎯 NEW: Real-time playback cursor shows current position!")
        print(f"🎯 Cursor smoothness: {self.cursor_fps} FPS ({self.cursor_sleep*1000:.1f}ms updates)")
        print("🐛 Debug: Watch console for cursor position updates")
        print("="*60)
        
        if not self.samples:
            print("❌ No samples to display!")
            return
        
        try:
            # Display first sample
            self.display_current_sample()
            
            # Setup keyboard shortcuts
            self.setup_keyboard_shortcuts()
            
            # Show the plot
            plt.show()
            
        except Exception as e:
            print(f"❌ Runtime error: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Enhanced Audio Analysis Player for Wav2Seg Predictions')
    parser.add_argument('--group', choices=['worst', 'best', 'all'], default='all',
                       help='Which group to analyze (default: all)')
    parser.add_argument('--auto-play', action='store_true',
                       help='Auto-play audio when loading samples')
    parser.add_argument('--dir', default='prediction_analysis',
                       help='Prediction analysis directory (default: prediction_analysis)')
    parser.add_argument('--smooth-cursor', action='store_true', default=True,
                       help='Use 60 FPS smooth cursor movement (default: True)')
    parser.add_argument('--standard-cursor', action='store_true',
                       help='Use 30 FPS standard cursor movement')
    
    args = parser.parse_args()
    
    # Determine cursor smoothness
    smooth_cursor = args.smooth_cursor and not args.standard_cursor
    
    # Check if directory exists
    if not os.path.exists(args.dir):
        print(f"❌ Directory '{args.dir}' not found!")
        print("Make sure you've run the wav2seg evaluation first.")
        sys.exit(1)
    
    # Create and run player
    try:
        player = AudioAnalysisPlayer(
            prediction_dir=args.dir,
            group=args.group,
            auto_play=args.auto_play,
            smooth_cursor=smooth_cursor
        )
        player.run()
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 