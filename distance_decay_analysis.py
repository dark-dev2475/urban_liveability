"""
Distance Decay Function Analysis and Visualization
This script demonstrates the mathematical behavior of the distance decay function
used in the urban liveability model.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def distance_decay_function(distance, alpha=0.25, context_factor=1.0, max_distance=5.0):
    """
    Calculate distance decay factor using exponential decay
    
    Args:
        distance: Distance in km
        alpha: Decay rate parameter (higher = faster decay)
        context_factor: Regional context adjustment
        max_distance: Maximum reasonable distance
    
    Returns:
        Distance factor (0-1 scale)
    """
    if distance <= max_distance:
        return np.exp(-alpha * distance * context_factor)
    else:
        return 0

def linear_decay_function(distance, max_distance=5.0):
    """
    Alternative: Linear decay function for comparison
    """
    if distance <= max_distance:
        return max(0, 1 - (distance / max_distance))
    else:
        return 0

def inverse_decay_function(distance, max_distance=5.0):
    """
    Alternative: Inverse decay function for comparison
    """
    if distance <= max_distance:
        return 1 / (1 + distance)
    else:
        return 0

# Create distance range for analysis
distances = np.linspace(0, 6, 100)

# Calculate decay values for different functions
exponential_decay = [distance_decay_function(d) for d in distances]
linear_decay = [linear_decay_function(d) for d in distances]
inverse_decay = [inverse_decay_function(d) for d in distances]

# Different alpha values for exponential decay
alpha_low = [distance_decay_function(d, alpha=0.1) for d in distances]
alpha_medium = [distance_decay_function(d, alpha=0.25) for d in distances]
alpha_high = [distance_decay_function(d, alpha=0.5) for d in distances]

# Different context factors
context_high_density = [distance_decay_function(d, context_factor=0.8) for d in distances]
context_medium_density = [distance_decay_function(d, context_factor=1.0) for d in distances]
context_low_density = [distance_decay_function(d, context_factor=1.2) for d in distances]

# Create visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Comparison of different decay functions
ax1.plot(distances, exponential_decay, label='Exponential Decay (α=0.25)', linewidth=2, color='blue')
ax1.plot(distances, linear_decay, label='Linear Decay', linewidth=2, color='red', linestyle='--')
ax1.plot(distances, inverse_decay, label='Inverse Decay', linewidth=2, color='green', linestyle=':')
ax1.set_xlabel('Distance (km)')
ax1.set_ylabel('Distance Factor')
ax1.set_title('Comparison of Different Decay Functions')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 6)
ax1.set_ylim(0, 1.1)

# Plot 2: Effect of different alpha values
ax2.plot(distances, alpha_low, label='α = 0.1 (Slow decay)', linewidth=2, color='green')
ax2.plot(distances, alpha_medium, label='α = 0.25 (Medium decay)', linewidth=2, color='blue')
ax2.plot(distances, alpha_high, label='α = 0.5 (Fast decay)', linewidth=2, color='red')
ax2.set_xlabel('Distance (km)')
ax2.set_ylabel('Distance Factor')
ax2.set_title('Effect of Alpha Parameter on Decay Rate')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 6)
ax2.set_ylim(0, 1.1)

# Plot 3: Effect of context factor (urban density)
ax3.plot(distances, context_high_density, label='High Density (CF=0.8)', linewidth=2, color='red')
ax3.plot(distances, context_medium_density, label='Medium Density (CF=1.0)', linewidth=2, color='blue')
ax3.plot(distances, context_low_density, label='Low Density (CF=1.2)', linewidth=2, color='green')
ax3.set_xlabel('Distance (km)')
ax3.set_ylabel('Distance Factor')
ax3.set_title('Effect of Urban Density on Distance Perception')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 6)
ax3.set_ylim(0, 1.1)

# Plot 4: Practical example with real amenities
# Simulate different amenities at various distances
amenity_distances = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
amenity_names = ['Local Shop', 'Bus Stop', 'Clinic', 'School', 'Hospital', 'Mall', 'University', 'Airport']
decay_values = [distance_decay_function(d) for d in amenity_distances]

ax4.bar(range(len(amenity_names)), decay_values, color=plt.cm.RdYlGn([v for v in decay_values]))
ax4.set_xlabel('Amenities')
ax4.set_ylabel('Accessibility Factor')
ax4.set_title('Practical Example: Amenity Accessibility by Distance')
ax4.set_xticks(range(len(amenity_names)))
ax4.set_xticklabels(amenity_names, rotation=45, ha='right')

# Add distance labels on bars
for i, (dist, val) in enumerate(zip(amenity_distances, decay_values)):
    ax4.text(i, val + 0.02, f'{dist}km\n{val:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('distance_decay_analysis.png', dpi=300, bbox_inches='tight')
print("Distance decay analysis saved as 'distance_decay_analysis.png'")

# Create a numerical comparison table
print("\n" + "="*60)
print("NUMERICAL COMPARISON OF DECAY FUNCTIONS")
print("="*60)

comparison_distances = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
print(f"{'Distance (km)':<12} {'Exponential':<12} {'Linear':<12} {'Inverse':<12}")
print("-" * 50)

for d in comparison_distances:
    exp_val = distance_decay_function(d)
    lin_val = linear_decay_function(d)
    inv_val = inverse_decay_function(d)
    print(f"{d:<12.1f} {exp_val:<12.3f} {lin_val:<12.3f} {inv_val:<12.3f}")

# Mathematical properties analysis
print("\n" + "="*60)
print("MATHEMATICAL PROPERTIES ANALYSIS")
print("="*60)

print("\n1. DECAY RATE ANALYSIS:")
print("   - At 1km: Factor =", f"{distance_decay_function(1.0):.3f}")
print("   - At 2km: Factor =", f"{distance_decay_function(2.0):.3f}")
print("   - At 3km: Factor =", f"{distance_decay_function(3.0):.3f}")
print("   - Relative impact of 1km vs 2km:", f"{distance_decay_function(1.0)/distance_decay_function(2.0):.2f}x")

print("\n2. HALF-LIFE CALCULATION:")
# Find distance where decay factor = 0.5
half_life_distance = -np.log(0.5) / 0.25
print(f"   - Distance where accessibility drops to 50%: {half_life_distance:.2f} km")

print("\n3. CONTEXT FACTOR IMPACT:")
for density, cf in [('High', 0.8), ('Medium', 1.0), ('Low', 1.2)]:
    factor_2km = distance_decay_function(2.0, context_factor=cf)
    print(f"   - {density} density (CF={cf}): 2km factor = {factor_2km:.3f}")

plt.show()
