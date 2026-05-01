#!/usr/bin/env python3

import json
import os
from collections import Counter

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from google import genai


class ModeClassifierNode(Node):
    def __init__(self):
        super().__init__('mode_classifier_node') #create ros2 node

        # set Parameters
        self.declare_parameter('model_name', 'gemini-2.5-flash')
        self.declare_parameter('num_votes', 3)

        self.model_name = self.get_parameter('model_name').value #read parameters
        self.num_votes = int(self.get_parameter('num_votes').value)

        # Gemini client
        api_key = os.environ.get('GEMINI_API_KEY') #read it from envi
        if not api_key:
            raise RuntimeError('GEMINI_API_KEY is not set.')

        self.client = genai.Client(api_key=api_key) #create gemini client

        # subs to /voice_trancript
        self.transcript_sub = self.create_subscription(
            String,
            '/voice_transcript',
            self.transcript_callback,
            10
        )

        self.raw_pub = self.create_publisher(String, '/emotion_state_raw', 10) #publis 1st raw classification result
        self.smoothed_pub = self.create_publisher(String, '/emotion_state_smoothed', 10) #publish smoothed

        self.get_logger().info('Mode classifier node started.')

    def build_prompt(self, transcript: str) -> str:
        return f"""
You are an emotion-mode classifier for a social robot.

Classify the user utterance into exactly one of these four labels:
- positive_high_energy
- negative_high_energy
- neutral
- negative_low_energy

Return JSON only in this exact schema:
{{
  "label": "...",
  "confidence": 0.0,
  "evidence": "short phrase from utterance"
}}

Rules:
- Choose exactly one label.
- Do not explain.
- Do not add markdown.
- Confidence must be between 0.0 and 1.0.

Utterance: "{transcript}"
""".strip()

    def classify_transcript(self, transcript: str) -> dict:
    #create prompt for current transcript
        prompt = self.build_prompt(transcript)

        response = self.client.models.generate_content( #send request to gemini
            model=self.model_name,
            contents=prompt,
        )
        text = response.text.strip() #get text response and parse it as JSON
        result = json.loads(text)

        label = result.get('label', 'neutral') #extract fields with fallback defaults
        confidence = float(result.get('confidence', 0.0))
        evidence = result.get('evidence', '')

    #allowed labels
        valid_labels = {
            'positive_high_energy',
            'negative_high_energy',
            'neutral',
            'negative_low_energy',
        }

        if label not in valid_labels:
            label = 'neutral'

        confidence = max(0.0, min(1.0, confidence))

        return {
            'transcript': transcript,
            'label': label,
            'confidence': confidence,
            'evidence': evidence,
        }


    #def smooth_label(self, new_label: str) -> str:
        #self.label_window.append(new_label)
        #counts = Counter(self.label_window)
        #top = counts.most_common()

        #if len(top) == 1:
            #self.prev_smoothed_label = top[0][0]
            #return self.prev_smoothed_label

        #if len(top) > 1 and top[0][1] == top[1][1]:
            #return self.prev_smoothed_label

        #self.prev_smoothed_label = top[0][0]
        #return self.prev_smoothed_label
        

    def smooth_label(self, transcript: str) -> dict:
        #run 3 times gemini calls and combine them by voting
        results = []

        self.get_logger().info('--- Voting start ---')
        self.get_logger().info(f'Transcript: {transcript}')
        self.get_logger().info(f'Number of Gemini calls: {self.num_votes}')
    #collect multiple classification results
        for i in range(self.num_votes):
            result = self.classify_transcript(transcript)
            results.append(result)

            self.get_logger().info(
                f"Vote {i + 1}/{self.num_votes}: "
                f"label={result['label']}, "
                f"confidence={result['confidence']:.3f}, "
                f"evidence={result['evidence']}"
            )
    #extract labels and count them
        labels = [r['label'] for r in results]
        counts = Counter(labels)
        top = counts.most_common()
        
        #debug logs

        for i, label in enumerate(labels):
            self.get_logger().info(f'Label[{i}] = {label}')

        for label, count in counts.items():
            self.get_logger().info(f'Count[{label}] = {count}')

        self.get_logger().info(f'Sorted counts: {top}')

        if len(top) == 1: 
            final_label = top[0][0]
            decision_reason = 'all_votes_same'
        elif len(counts) == len(labels):
            best = max(results, key=lambda r: r['confidence'])
            final_label = best['label']
            decision_reason = 'all_votes_different_highest_confidence'
        #elif len(top) > 1 and top[0][1] == top[1][1]:
            #tied_labels = [item[0] for item in top if item[1] == top[0][1]]
            #tied_results = [r for r in results if r['label'] in tied_labels]
            #best = max(tied_results, key=lambda r: r['confidence'])
            #final_label = best['label']
            #decision_reason = 'tie_highest_confidence'
        else:
            final_label = top[0][0]
            decision_reason = 'majority_vote'

        self.get_logger().info(f'Final label: {final_label}')
        self.get_logger().info(f'Decision reason: {decision_reason}')
        self.get_logger().info('--- Voting end ---')

        return {
            'transcript': transcript,
            'final_label': final_label,
            'decision_reason': decision_reason,
            'all_results': results,
            'labels': labels,
            'counts': dict(counts),
        }

    def transcript_callback(self, msg: String):
        transcript = msg.data.strip()
        if not transcript:
            return

        try:
            voted_result = self.smooth_label(transcript)

            raw_msg = String()
            raw_msg.data = json.dumps(voted_result['all_results'][0])
            self.raw_pub.publish(raw_msg)

            smooth_msg = String()
            smooth_msg.data = json.dumps(voted_result)
            self.smoothed_pub.publish(smooth_msg)

            self.get_logger().info(
                f"First raw label: {voted_result['all_results'][0]['label']}"
            )
            self.get_logger().info(
                f"Final voted label: {voted_result['final_label']}"
            )

        except Exception as e:
            self.get_logger().error(f'Classification failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ModeClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
