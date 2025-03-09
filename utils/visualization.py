import graphviz
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from io import BytesIO
import base64
from typing import Dict, List, Any
import numpy as np

class ContentVisualizer:
    def __init__(self):
        self.colors = {
            'main_topic': '#7792E3',
            'subtopic': '#4CAF50',
            'key_concept': '#FF9800',
            'connection': '#9C27B0'
        }

    def generate_mindmap(self, topics: Dict[str, List[str]], highlighted_terms: Dict[str, str]) -> str:
        """Generate a mindmap visualization of topics and their relationships."""
        dot = graphviz.Digraph(comment='Content Mindmap')
        dot.attr(rankdir='LR')

        # Set default node attributes
        dot.attr('node', shape='box', style='rounded,filled', fillcolor='white')

        # Add main topics
        for topic, subtopics in topics.items():
            # Add main topic node
            dot.node(topic, topic, fillcolor=self.colors['main_topic'], fontcolor='white')

            # Add subtopics
            for subtopic in subtopics:
                dot.node(subtopic, subtopic, fillcolor=self.colors['subtopic'], fontcolor='white')
                dot.edge(topic, subtopic)

            # Add related terms
            related_terms = [term for term in highlighted_terms.keys() if term in ' '.join(subtopics)]
            for term in related_terms:
                dot.node(term, term, fillcolor=self.colors['key_concept'], fontcolor='white')
                # Connect to relevant subtopics
                for subtopic in subtopics:
                    if term in subtopic:
                        dot.edge(subtopic, term, color=self.colors['connection'])

        # Return SVG as string
        return dot.pipe(format='svg').decode('utf-8')

    def create_concept_hierarchy(self, topics: Dict[str, List[str]], highlight_terms: Dict[str, str]) -> str:
        """Create a hierarchical visualization of concepts."""
        # Create figure and axis
        fig = Figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        # Calculate positions
        num_topics = len(topics)
        topic_positions = np.linspace(0, 1, num_topics)
        
        for idx, (topic, subtopics) in enumerate(topics.items()):
            # Plot main topic
            y_pos = 1.0
            x_pos = topic_positions[idx]
            ax.plot([x_pos], [y_pos], 'o', color=self.colors['main_topic'], markersize=15)
            ax.annotate(topic, (x_pos, y_pos), xytext=(5, 5), textcoords='offset points')

            # Plot subtopics
            num_subtopics = len(subtopics)
            if num_subtopics > 0:
                subtopic_x = np.linspace(x_pos - 0.1, x_pos + 0.1, num_subtopics)
                subtopic_y = [0.7] * num_subtopics
                
                for sx, sy, subtopic in zip(subtopic_x, subtopic_y, subtopics):
                    ax.plot([x_pos, sx], [y_pos, sy], '-', color=self.colors['subtopic'])
                    ax.plot([sx], [sy], 'o', color=self.colors['subtopic'], markersize=10)
                    ax.annotate(subtopic, (sx, sy), xytext=(5, -15), textcoords='offset points')

                    # Add related terms
                    related_terms = [term for term, _ in highlight_terms.items() if term in subtopic]
                    if related_terms:
                        term_x = np.linspace(sx - 0.05, sx + 0.05, len(related_terms))
                        term_y = [0.4] * len(related_terms)
                        
                        for tx, ty, term in zip(term_x, term_y, related_terms):
                            ax.plot([sx, tx], [sy, ty], '--', color=self.colors['connection'])
                            ax.plot([tx], [ty], 's', color=self.colors['key_concept'], markersize=8)
                            ax.annotate(term, (tx, ty), xytext=(5, -15), textcoords='offset points')

        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(0.3, 1.1)
        ax.axis('off')
        fig.tight_layout()

        # Convert to SVG
        buf = BytesIO()
        fig.savefig(buf, format='svg', bbox_inches='tight')
        buf.seek(0)
        return buf.getvalue().decode('utf-8')

    def generate_learning_infographic(self, content_stats: Dict[str, Any]) -> str:
        """Generate an infographic showing learning statistics and patterns."""
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        # Extract metrics
        metrics = {
            'Completion Rate': content_stats.get('completion_rate', 0) * 100,
            'Engagement Score': content_stats.get('engagement_score', 0) * 100,
            'Understanding Level': content_stats.get('understanding_level', 0) * 100
        }

        # Create bar chart
        bars = ax.bar(range(len(metrics)), list(metrics.values()))
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics.keys(), rotation=45)

        # Customize appearance
        ax.set_ylim(0, 100)
        ax.set_ylabel('Percentage')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}%',
                   ha='center', va='bottom')

        # Style the chart
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()

        # Convert to SVG
        buf = BytesIO()
        fig.savefig(buf, format='svg', bbox_inches='tight')
        buf.seek(0)
        return buf.getvalue().decode('utf-8')
