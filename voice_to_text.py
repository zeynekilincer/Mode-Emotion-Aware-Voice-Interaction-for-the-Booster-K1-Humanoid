#!/usr/bin/env python3

import numpy as np
import sounddevice as sd
import whisper

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class VoiceTranscriptNode(Node):
    def __init__(self):
        super().__init__('voice_transcript_node') # ros2 node

        self.transcript_pub = self.create_publisher(String, '/voice_transcript', 10) #publisher publishing text to topic voice_trans

        self.declare_parameter('model_size', 'base') #set default parametrs
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('chunk_seconds', 4.0)
        self.declare_parameter('energy_threshold', 0.01)

        self.model_size = self.get_parameter('model_size').value #read parametrs
        self.sample_rate = int(self.get_parameter('sample_rate').value)
        self.chunk_seconds = float(self.get_parameter('chunk_seconds').value)
        self.energy_threshold = float(self.get_parameter('energy_threshold').value)

        self.get_logger().info(f'Loading Whisper model: {self.model_size}') #load whisper once when node starts
        self.model = whisper.load_model(self.model_size)
        self.get_logger().info('Whisper model loaded.')
        self.get_logger().info('Voice transcript node ready.')


    def record_one_utterance(self): #record one fixed length audio chunk ftom default microphone
        self.get_logger().info('Recording one utterance...')
        audio = sd.rec(
            int(self.chunk_seconds * self.sample_rate), #number of samples
            samplerate=self.sample_rate, #samples per sec, mono audio, float samples for whisper
            channels=1,
            dtype='float32'
        )
        sd.wait() #wait until recording finishes

        audio = np.squeeze(audio) #remove extra dimension
        rms = np.sqrt(np.mean(np.square(audio))) if len(audio) > 0 else 0.0 # to detect silence

        if rms < self.energy_threshold:
            self.get_logger().info('Audio too quiet, skipping.')
            return None

        return audio

    def transcribe_audio(self, audio_np): #send recorded nmpy audio directly to whisper
        result = self.model.transcribe(audio_np, fp16=False, language='en')
        return result.get("text", "").strip() #return only text part of the output
    def publish_transcript(self, transcript):
        msg = String() #create ros2 string msg to publsih
        msg.data = transcript
        self.transcript_pub.publish(msg)
        self.get_logger().info(f'Transcript: {transcript}')


    def run_turn_based_loop(self): #wait for enter
        while rclpy.ok():
            try:
                input("\nPress Enter to record one utterance, or Ctrl+C to quit...")

                audio = self.record_one_utterance() #record
                if audio is None:
                    continue

                transcript = self.transcribe_audio(audio)
                if not transcript.strip():
                    self.get_logger().info('Empty transcript.')
                    continue

                self.publish_transcript(transcript)

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.get_logger().error(f'Processing failed: {e}')


def main(args=None):
    rclpy.init(args=args) #initialize
    node = VoiceTranscriptNode() #start the node

    try:
        node.run_turn_based_loop() #run loop
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


