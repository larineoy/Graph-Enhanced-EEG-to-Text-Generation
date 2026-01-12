"""
Generate architecture diagram using matplotlib
Shows STRG → STRE → Decoder pipeline
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np
import argparse
import os


def draw_architecture_diagram(
    output_path: str = 'checkpoints/architecture_diagram.png',
    figsize: tuple = (16, 10),
    dpi: int = 300
):
    """
    Draw architecture diagram showing the model pipeline
    
    Args:
        output_path: Path to save figure
        figsize: Figure size
        dpi: Resolution
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    colors = {
        'input': '#E8F4F8',  # Light blue
        'strg': '#FFE5B4',  # Light orange
        'stre': '#D4EDDA',  # Light green
        'decoder': '#F8D7DA',  # Light red
        'output': '#E8F4F8',  # Light blue
        'text': '#F5F5F5'  # Light gray
    }
    
    border_colors = {
        'input': '#2E86AB',
        'strg': '#F18F01',
        'stre': '#6A994E',
        'decoder': '#C73E1D',
        'output': '#2E86AB',
        'text': '#666666'
    }
    
    # Box dimensions
    box_width = 2.5
    box_height = 1.2
    v_spacing = 1.5
    h_spacing = 1.0
    
    # Define positions
    positions = {
        'eeg_input': (1, 8),
        'text_input': (1, 6),
        
        'strg': (4.5, 7),
        'strg_components': (4.5, 5.5),
        
        'stre': (7, 7),
        
        'decoder': (4.5, 3),
        'text_output': (7, 3),
    }
    
    # 1. Input boxes
    # EEG Input
    eeg_box = FancyBboxPatch(
        positions['eeg_input'], box_width, box_height,
        boxstyle="round,pad=0.1", facecolor=colors['input'],
        edgecolor=border_colors['input'], linewidth=2
    )
    ax.add_patch(eeg_box)
    ax.text(positions['eeg_input'][0] + box_width/2, positions['eeg_input'][1] + box_height/2,
           'EEG Signal\n(Frequency Bands)', ha='center', va='center',
           fontsize=11, fontweight='bold')
    
    # Text Input
    text_box = FancyBboxPatch(
        positions['text_input'], box_width, box_height,
        boxstyle="round,pad=0.1", facecolor=colors['text'],
        edgecolor=border_colors['text'], linewidth=2
    )
    ax.add_patch(text_box)
    ax.text(positions['text_input'][0] + box_width/2, positions['text_input'][1] + box_height/2,
           'Text Tokens\n(Training)', ha='center', va='center',
           fontsize=11, fontweight='bold')
    
    # 2. STRG Module
    strg_box = FancyBboxPatch(
        positions['strg'], box_width, box_height,
        boxstyle="round,pad=0.1", facecolor=colors['strg'],
        edgecolor=border_colors['strg'], linewidth=2.5
    )
    ax.add_patch(strg_box)
    ax.text(positions['strg'][0] + box_width/2, positions['strg'][1] + box_height/2,
           'STRG\n(Spectro-Topographic\nRelational Graph)', ha='center', va='center',
           fontsize=11, fontweight='bold')
    
    # STRG Components
    strg_comp_box = FancyBboxPatch(
        positions['strg_components'], box_width, box_height * 0.8,
        boxstyle="round,pad=0.05", facecolor=colors['strg'],
        edgecolor=border_colors['strg'], linewidth=1.5, alpha=0.7
    )
    ax.add_patch(strg_comp_box)
    ax.text(positions['strg_components'][0] + box_width/2, positions['strg_components'][1] + box_height * 0.4,
           'Spatial + Functional\nConnectivity', ha='center', va='center',
           fontsize=9, style='italic')
    
    # 3. STRE Module
    stre_box = FancyBboxPatch(
        positions['stre'], box_width, box_height,
        boxstyle="round,pad=0.1", facecolor=colors['stre'],
        edgecolor=border_colors['stre'], linewidth=2.5
    )
    ax.add_patch(stre_box)
    ax.text(positions['stre'][0] + box_width/2, positions['stre'][1] + box_height/2,
           'STRE\n(Spatio-Temporal\nRelational Embeddings)', ha='center', va='center',
           fontsize=11, fontweight='bold')
    
    # 4. Decoder
    decoder_box = FancyBboxPatch(
        positions['decoder'], box_width * 1.2, box_height,
        boxstyle="round,pad=0.1", facecolor=colors['decoder'],
        edgecolor=border_colors['decoder'], linewidth=2.5
    )
    ax.add_patch(decoder_box)
    ax.text(positions['decoder'][0] + box_width * 0.6, positions['decoder'][1] + box_height/2,
           'Transformer\nDecoder', ha='center', va='center',
           fontsize=11, fontweight='bold')
    
    # 5. Output
    output_box = FancyBboxPatch(
        positions['text_output'], box_width, box_height,
        boxstyle="round,pad=0.1", facecolor=colors['output'],
        edgecolor=border_colors['output'], linewidth=2
    )
    ax.add_patch(output_box)
    ax.text(positions['text_output'][0] + box_width/2, positions['text_output'][1] + box_height/2,
           'Generated\nText', ha='center', va='center',
           fontsize=11, fontweight='bold')
    
    # Draw arrows
    arrow_props = dict(arrowstyle='->', lw=2.5, color='#333333')
    
    # EEG → STRG
    arrow1 = FancyArrowPatch(
        (positions['eeg_input'][0] + box_width, positions['eeg_input'][1] + box_height/2),
        (positions['strg'][0], positions['strg'][1] + box_height/2),
        **arrow_props
    )
    ax.add_patch(arrow1)
    
    # STRG → STRE
    arrow2 = FancyArrowPatch(
        (positions['strg'][0] + box_width, positions['strg'][1] + box_height/2),
        (positions['stre'][0], positions['stre'][1] + box_height/2),
        **arrow_props
    )
    ax.add_patch(arrow2)
    
    # STRE → Decoder
    arrow3 = FancyArrowPatch(
        (positions['stre'][0] + box_width/2, positions['stre'][1]),
        (positions['decoder'][0] + box_width * 0.6, positions['decoder'][1] + box_height),
        connectionstyle="arc3,rad=0.2", **arrow_props
    )
    ax.add_patch(arrow3)
    
    # Text Input → Decoder (for teacher forcing)
    arrow4 = FancyArrowPatch(
        (positions['text_input'][0] + box_width, positions['text_input'][1] + box_height/2),
        (positions['decoder'][0], positions['decoder'][1] + box_height/2),
        connectionstyle="arc3,rad=-0.3", **arrow_props, linestyle='--', alpha=0.6
    )
    ax.add_patch(arrow4)
    ax.text(positions['text_input'][0] + box_width + 0.5, positions['text_input'][1] - 0.3,
           'Teacher\nForcing', ha='left', va='top', fontsize=8, style='italic', alpha=0.7)
    
    # Decoder → Output
    arrow5 = FancyArrowPatch(
        (positions['decoder'][0] + box_width * 1.2, positions['decoder'][1] + box_height/2),
        (positions['text_output'][0], positions['text_output'][1] + box_height/2),
        **arrow_props
    )
    ax.add_patch(arrow5)
    
    # Add title
    ax.text(5, 9.5, 'Graph-Enhanced EEG-to-Text Generation Architecture', 
           ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['input'], edgecolor=border_colors['input'], label='Input/Output'),
        mpatches.Patch(facecolor=colors['strg'], edgecolor=border_colors['strg'], label='STRG Module'),
        mpatches.Patch(facecolor=colors['stre'], edgecolor=border_colors['stre'], label='STRE Module'),
        mpatches.Patch(facecolor=colors['decoder'], edgecolor=border_colors['decoder'], label='Transformer Decoder'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved architecture diagram to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate architecture diagram')
    parser.add_argument('--output', type=str, default='checkpoints/architecture_diagram.png',
                       help='Path to save figure')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Figure resolution (DPI)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("GENERATING ARCHITECTURE DIAGRAM")
    print("="*70)
    
    draw_architecture_diagram(
        output_path=args.output,
        dpi=args.dpi
    )
    
    print("\n" + "="*70)
    print("ARCHITECTURE DIAGRAM GENERATED!")
    print("="*70)


if __name__ == '__main__':
    main()
