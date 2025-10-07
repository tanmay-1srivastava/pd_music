#!/usr/bin/env python3
"""
Visual Comparison of Gait Parameters: Expensive Sensors vs Inexpensive IMU
Analysis for Marcy - July 28th Session 1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

def create_visual_comparison():
    """Create visual bar plot comparison between expensive and inexpensive sensors"""
    
    print("Creating visual comparison plots for Marcy...")
    
    # Load the latest results from inexpensive IMU
    with open('Results/individual_sessions/Marcy_July_28_Session_1.json', 'r') as f:
        imu_data = json.load(f)
    
    # Extract cheap sensor data (updated results)
    cheap_left = imu_data['cheap_sensor']['left_foot']
    cheap_right = imu_data['cheap_sensor']['right_foot']
    cheap_combined = imu_data['cheap_sensor']['combined']
    
    # Data from expensive sensors (from your table for Marcy July 28th)
    expensive_data = {
        "Cadence (steps/min)": 107,  # Average of L&R (107|107)
        "Gait Speed (m/s)": 1.13,   # Average of L&R (1.14|1.12)
        "Step Duration (s)": 0.562,  # Average of L&R (0.537|0.587) 
        "Stride Length (m)": 1.27,   # Average of L&R (1.28|1.26)
        "Stride Time (s)": 1.12,     # Average of L&R (1.12|1.12)
        "Double Support (s)": 0.274  # Calculated from %GCT (24.3% of 1.12s)
    }
    
    # Calculate corresponding values from cheap IMU
    cheap_cadence_left = 60 / cheap_left['step_time_stats']['mean']
    cheap_cadence_right = 60 / cheap_right['step_time_stats']['mean'] 
    cheap_cadence = (cheap_cadence_left + cheap_cadence_right) / 2
    
    cheap_step_time = (cheap_left['step_time_stats']['mean'] + cheap_right['step_time_stats']['mean']) / 2
    cheap_stride_length = (cheap_left['stride_length_stats']['mean'] + cheap_right['stride_length_stats']['mean']) / 2
    cheap_stride_time = (cheap_left['stride_time_stats']['mean'] + cheap_right['stride_time_stats']['mean']) / 2
    cheap_gait_speed = cheap_combined['gait_velocity_stats']['mean']
    cheap_double_support = cheap_combined['double_support_stats']['mean']
    
    cheap_data = {
        "Cadence (steps/min)": cheap_cadence,
        "Gait Speed (m/s)": cheap_gait_speed,
        "Step Duration (s)": cheap_step_time,
        "Stride Length (m)": cheap_stride_length,
        "Stride Time (s)": cheap_stride_time,
        "Double Support (s)": cheap_double_support
    }
    
    # Create the comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Gait Parameter Comparison: Expensive vs Inexpensive Sensors\nMarcy - July 28th Session 1', 
                 fontsize=16, fontweight='bold')
    
    parameters = list(expensive_data.keys())
    colors = ['#1f77b4', '#ff7f0e']  # Blue for expensive, orange for cheap
    
    for i, param in enumerate(parameters):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Data for this parameter
        expensive_val = expensive_data[param]
        cheap_val = cheap_data[param]
        
        # Create bar plot
        x_pos = [0, 1]
        values = [expensive_val, cheap_val]
        labels = ['Expensive\nSensors', 'Inexpensive\nIMU']
        
        bars = ax.bar(x_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Customize the plot
        ax.set_title(param, fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for j, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{val:.3f}' if val < 10 else f'{val:.1f}',
                   ha='center', va='bottom', fontweight='bold')
        
        # Calculate and display difference
        diff = abs(expensive_val - cheap_val)
        diff_percent = (diff / expensive_val) * 100
        
        # Add difference annotation
        ax.text(0.5, max(values) * 0.85, 
               f'Diff: {diff:.3f}\n({diff_percent:.1f}%)',
               ha='center', va='center', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
               fontsize=10)
        
        # Set y-axis to start from 0
        ax.set_ylim(0, max(values) * 1.15)
    
    plt.tight_layout()
    plt.savefig('Results/plots/Marcy_Parameter_Comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a summary table
    print("\n" + "="*80)
    print("QUANTITATIVE COMPARISON SUMMARY - MARCY")
    print("="*80)
    
    comparison_df = pd.DataFrame({
        'Parameter': parameters,
        'Expensive Sensors': [expensive_data[p] for p in parameters],
        'Inexpensive IMU': [cheap_data[p] for p in parameters],
        'Absolute Difference': [abs(expensive_data[p] - cheap_data[p]) for p in parameters],
        'Percent Difference': [(abs(expensive_data[p] - cheap_data[p]) / expensive_data[p]) * 100 for p in parameters]
    })
    
    # Format the display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.3f}'.format)
    
    print(comparison_df.to_string(index=False))
    
    # Accuracy assessment
    print("\n" + "="*80)
    print("ACCURACY ASSESSMENT - MARCY")
    print("="*80)
    
    # Calculate overall accuracy metrics
    mean_percent_error = comparison_df['Percent Difference'].mean()
    
    # Categorize accuracy
    good_params = comparison_df[comparison_df['Percent Difference'] < 10]['Parameter'].tolist()
    moderate_params = comparison_df[(comparison_df['Percent Difference'] >= 10) & 
                                   (comparison_df['Percent Difference'] < 30)]['Parameter'].tolist()
    poor_params = comparison_df[comparison_df['Percent Difference'] >= 30]['Parameter'].tolist()
    
    print(f"Overall Mean Percent Error: {mean_percent_error:.1f}%")
    print(f"\nGOOD Agreement (<10% error): {len(good_params)} parameters")
    for param in good_params:
        error = comparison_df[comparison_df['Parameter'] == param]['Percent Difference'].iloc[0]
        print(f"  ✓ {param}: {error:.1f}% error")
    
    print(f"\nMODERATE Agreement (10-30% error): {len(moderate_params)} parameters")
    for param in moderate_params:
        error = comparison_df[comparison_df['Parameter'] == param]['Percent Difference'].iloc[0]
        print(f"  ⚠ {param}: {error:.1f}% error")
    
    print(f"\nPOOR Agreement (>30% error): {len(poor_params)} parameters")
    for param in poor_params:
        error = comparison_df[comparison_df['Parameter'] == param]['Percent Difference'].iloc[0]
        print(f"  ✗ {param}: {error:.1f}% error")
    
    # Save comparison data
    comparison_df.to_csv('Results/summary/Marcy_Parameter_Comparison.csv', index=False)
    
    return comparison_df

if __name__ == "__main__":
    comparison_results = create_visual_comparison()