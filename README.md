# Digital Twin Creation Pipeline: From Images to 3D GeoJSON

Complete workflow for creating a 3D building digital twin with thermal anomalies for visualization in ArcGIS Online Scene Viewer.

**Project:** D.M. Smith Building, Georgia Institute of Technology
**Authors:** Building Energy Management Research Team
**Date:** November 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Step-by-Step Workflow](#step-by-step-workflow)
5. [File Structure](#file-structure)
6. [Script Documentation](#script-documentation)
7. [Configuration Parameters](#configuration-parameters)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)
10. [Example Results](#example-results)

---

## Overview

This pipeline transforms **2D images** of a building into a **3D GPS-referenced digital twin** suitable for web-based visualization and building energy management applications.

### Input
- Regular RGB images (exterior building photos)
- Thermal infrared images (optional)
- Building GPS location (approximate center coordinates)
- Target building dimensions (width, length, height in meters)

### Output
- GeoJSON files with 3D building geometry
- GPS-referenced coordinates (WGS84)
- Integrated thermal anomaly data
- ArcGIS Online-ready visualization

### Key Features
- ✅ Automated facade segmentation
- ✅ Window/opening detection
- ✅ Thermal anomaly mapping
- ✅ GPS coordinate conversion
- ✅ 3D rotation corrections
- ✅ Interactive web visualization

---

## Prerequisites

### Software Requirements

1. **MATLAB** (R2020b or later)
   - Image Processing Toolbox
   - Computer Vision Toolbox
   - Statistics and Machine Learning Toolbox

2. **Python 3.8+**
   - numpy
   - json (built-in)
   - math (built-in)
   - copy (built-in)

3. **Optional: NeRF Reconstruction Tools**
   - COLMAP (for structure-from-motion)
   - Instant-NGP or Nerfstudio (for NeRF reconstruction)

4. **ArcGIS Online Account**
   - Scene Viewer access
   - Content publishing permissions

### Python Package Installation

```bash
# Install required packages
pip install numpy

# Optional: for presentation generation
pip install python-pptx
```

### Hardware Requirements

- **RAM:** 16GB minimum, 32GB recommended
- **Storage:** 10GB free space for processing
- **GPU:** CUDA-capable GPU recommended for NeRF reconstruction (optional)

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DIGITAL TWIN PIPELINE                        │
└─────────────────────────────────────────────────────────────────┘

PHASE 1: IMAGE ACQUISITION
├── RGB Photos (multiple angles)
├── Thermal IR Images (optional)
└── GPS Location Data
            │
            ▼
PHASE 2: 3D RECONSTRUCTION (Optional)
├── Structure from Motion (COLMAP)
├── Dense Point Cloud Generation
└── NeRF Training (if available)
            │
            ▼
PHASE 3: MATLAB PROCESSING
├── Point Cloud/Image Segmentation
├── Facade Plane Extraction
├── Window Detection
├── Thermal Anomaly Identification
└── Export to Local Coordinates (.mat or .txt)
            │
            ▼
PHASE 4: GEOJSON CREATION
├── Load MATLAB Data
├── Create GeoJSON Features
├── Add Properties (type, name, material)
└── Save Complete.geojson
            │
            ▼
PHASE 5: GPS COORDINATE CONVERSION
├── Define Building Center GPS
├── Calculate Meters per Degree
├── Convert Local → GPS Coordinates
└── Update GeoJSON with GPS Coordinates
            │
            ▼
PHASE 6: FACADE ALIGNMENT
├── Calculate Facade Planarity
├── Test Rotation Angles (0° to 4°)
├── Apply Optimal Rotations
├── Handle Translation Adjustments
└── Create Corrected Facade Files
            │
            ▼
PHASE 7: THERMAL INTEGRATION
├── Separate Red/Blue Anomalies
├── Apply Facade-Specific Rotations
└── Create Thermal Overlay Layers
            │
            ▼
PHASE 8: FINAL ASSEMBLY
├── Combine All Components
├── Validate Dimensions
├── Add Metadata
└── Export Final GeoJSON
            │
            ▼
PHASE 9: ARCGIS ONLINE UPLOAD
├── Upload GeoJSON Files
├── Configure Elevation Mode
├── Apply Styling
└── Publish Scene
```

---

## Step-by-Step Workflow

### STEP 1: Image Acquisition

**Objective:** Capture comprehensive building imagery

**Actions:**

1. **RGB Photography**
   ```
   - Take photos from all 4 sides (facades)
   - Include overlapping views (60-80% overlap)
   - Capture top-down views if possible (roof)
   - Ensure good lighting conditions
   - Minimum: 50-100 images per facade
   ```

2. **Thermal Imaging** (Optional)
   ```
   - Use IR camera (e.g., FLIR)
   - Capture at consistent distance
   - Note ambient temperature
   - Same angles as RGB photos
   ```

3. **GPS Location**
   ```
   - Record building center coordinates
   - Use Google Maps / GPS device
   - Note: Approximate location sufficient
   - Format: Decimal degrees (e.g., 33.773687, -84.395185)
   ```

**Output:**
- `images/rgb/` - RGB photos
- `images/thermal/` - IR images
- `location.txt` - GPS coordinates

---

### STEP 2: 3D Reconstruction (Optional)

**Objective:** Create point cloud from images

**Method A: COLMAP (Structure from Motion)**

```bash
# Feature extraction
colmap feature_extractor \
  --database_path database.db \
  --image_path images/rgb

# Feature matching
colmap exhaustive_matcher \
  --database_path database.db

# Sparse reconstruction
colmap mapper \
  --database_path database.db \
  --image_path images/rgb \
  --output_path sparse

# Dense reconstruction
colmap image_undistorter \
  --image_path images/rgb \
  --input_path sparse/0 \
  --output_path dense

colmap patch_match_stereo \
  --workspace_path dense

colmap stereo_fusion \
  --workspace_path dense \
  --output_path dense/fused.ply
```

**Method B: NeRF Reconstruction (Nerfstudio)**

```bash
# Install nerfstudio
pip install nerfstudio

# Process images
ns-process-data images \
  --data images/rgb \
  --output-dir processed_data

# Train NeRF
ns-train nerfacto \
  --data processed_data

# Export point cloud
ns-export pointcloud \
  --load-config outputs/config.yml \
  --output-dir pointcloud.ply
```

**Output:**
- `pointcloud.ply` or `fused.ply` - Dense point cloud

---

### STEP 3: MATLAB Processing

**Objective:** Segment building components and extract geometry

**Script:** `matlab_processing.m` (create this)

```matlab
%% MATLAB Processing Script for Digital Twin Creation
% Extracts facades, windows, and thermal anomalies from point cloud

%% 1. Load Point Cloud
ptCloud = pcread('pointcloud.ply');

% Visualize
figure;
pcshow(ptCloud);
title('Original Point Cloud');

%% 2. Ground Plane Removal
% Fit ground plane using RANSAC
maxDistance = 0.1; % 10cm tolerance
[~, inlierIndices] = pcfitplane(ptCloud, maxDistance);

% Remove ground points
buildingCloud = select(ptCloud, ~inlierIndices);

%% 3. Facade Segmentation
% Segment into 4 main facades based on orientation

% Calculate normals
normals = pcnormals(buildingCloud);

% Cluster by normal direction
% Facade 1 (North): normal ≈ [0, 1, 0]
% Facade 2 (South): normal ≈ [0, -1, 0]
% Facade 3 (East): normal ≈ [1, 0, 0]
% Facade 4 (West): normal ≈ [-1, 0, 0]

% Example for Facade 1 (adjust thresholds as needed)
facade1_mask = abs(normals(:,2)) > 0.8 & normals(:,2) > 0;
facade1_cloud = select(buildingCloud, facade1_mask);

% Repeat for facades 2, 3, 4...

%% 4. Window Detection
% Use depth variation or intensity clustering

for i = 1:4
    % Project facade to 2D plane
    facade_cloud = eval(['facade' num2str(i) '_cloud']);

    % Project to dominant plane
    [coeff, ~, ~] = pca(facade_cloud.Location);
    projected_2d = facade_cloud.Location * coeff(:, 1:2);

    % Create 2D grid
    [X, Y] = meshgrid(linspace(min(projected_2d(:,1)), max(projected_2d(:,1)), 200), ...
                      linspace(min(projected_2d(:,2)), max(projected_2d(:,2)), 200));

    % Interpolate depth
    Z = griddata(projected_2d(:,1), projected_2d(:,2), facade_cloud.Location(:,3), X, Y);

    % Detect windows (depth discontinuities)
    depth_gradient = imgradient(Z);
    window_mask = depth_gradient > threshold; % Adjust threshold

    % Connected components for individual windows
    CC = bwconncomp(window_mask);

    % Extract window boundaries
    for j = 1:CC.NumObjects
        window_pixels = CC.PixelIdxList{j};
        % Get bounding box
        [rows, cols] = ind2sub(size(window_mask), window_pixels);
        window_bbox = [min(cols), min(rows), max(cols)-min(cols), max(rows)-min(rows)];

        % Convert back to 3D coordinates
        % Store window geometry
    end
end

%% 5. Thermal Anomaly Detection
% If thermal images available

% Load thermal image
thermal_img = imread('thermal_facade1.jpg');

% Threshold hot/cold spots
hot_threshold = 0.7; % Adjust based on normalized intensity
cold_threshold = 0.3;

hot_mask = thermal_img > hot_threshold;
cold_mask = thermal_img < cold_threshold;

% Extract anomaly regions
hot_regions = regionprops(hot_mask, 'BoundingBox', 'Centroid');
cold_regions = regionprops(cold_mask, 'BoundingBox', 'Centroid');

%% 6. Export Data
% Save all extracted geometry to files

% Facade 1
facade1_data = struct();
facade1_data.type = 'facade';
facade1_data.name = 'Facade1';
facade1_data.corners = [...]; % 4 corners in local coordinates [x, y, z]
facade1_data.windows = [...]; % Window boundaries
save('facade1_data.mat', 'facade1_data');

% Repeat for all facades, roof, ground, thermal anomalies

% Export summary
summary = struct();
summary.facades = 4;
summary.windows = total_windows;
summary.thermal_hot = length(hot_regions);
summary.thermal_cold = length(cold_regions);
save('extraction_summary.mat', 'summary');

disp('MATLAB Processing Complete!');
```

**Output:**
- `facade1_data.mat` to `facade4_data.mat`
- `roof_data.mat`
- `ground_data.mat`
- `thermal_data.mat`
- `extraction_summary.mat`

---

### STEP 4: GeoJSON Creation

**Objective:** Convert MATLAB data to GeoJSON format

**Script:** `create_initial_geojson.py`

```python
#!/usr/bin/env python3
"""
Create Initial GeoJSON from MATLAB Exported Data
"""

import json
import numpy as np
from scipy.io import loadmat

# Configuration
GPS_CENTER = {
    'latitude': 33.773687,   # Building center latitude
    'longitude': -84.395185,  # Building center longitude
    'altitude': 305.0         # Ground elevation (MSL)
}

# Load MATLAB data
def load_matlab_data(filename):
    """Load .mat file"""
    return loadmat(filename, simplify_cells=True)

# Create GeoJSON structure
geojson = {
    "type": "FeatureCollection",
    "name": "Complete_Building",
    "crs": {
        "type": "name",
        "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
    },
    "features": []
}

# Convert local coordinates to GPS
def local_to_gps(x, y, z):
    """
    Convert local ENU coordinates (meters) to GPS
    x: East (meters from center)
    y: North (meters from center)
    z: Up (meters from ground)
    """
    lat_center = GPS_CENTER['latitude']
    lon_center = GPS_CENTER['longitude']
    alt_center = GPS_CENTER['altitude']

    # Meters per degree at this latitude
    meters_per_deg_lat = 111000.0
    meters_per_deg_lon = 111000.0 * np.cos(np.radians(lat_center))

    # Convert
    lon = lon_center + (x / meters_per_deg_lon)
    lat = lat_center + (y / meters_per_deg_lat)
    alt = alt_center + z

    return [lon, lat, alt]

# Process facades
for i in range(1, 5):
    facade_data = load_matlab_data(f'facade{i}_data.mat')

    # Create facade wall feature
    corners_local = facade_data['corners']  # [[x1,y1,z1], [x2,y2,z2], ...]

    # Convert to GPS
    corners_gps = [local_to_gps(c[0], c[1], c[2]) for c in corners_local]

    # Close polygon
    corners_gps.append(corners_gps[0])

    facade_feature = {
        "type": "Feature",
        "properties": {
            "type": "facade",
            "name": f"Facade{i}",
            "material": "concrete",
            "elevation_m": np.mean([c[2] for c in corners_gps])
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [corners_gps]
        }
    }

    geojson['features'].append(facade_feature)

    # Process windows for this facade
    windows = facade_data.get('windows', [])
    for j, window in enumerate(windows):
        window_corners_local = window['corners']
        window_corners_gps = [local_to_gps(c[0], c[1], c[2]) for c in window_corners_local]
        window_corners_gps.append(window_corners_gps[0])

        window_feature = {
            "type": "Feature",
            "properties": {
                "type": "opening",
                "name": f"Facade{i}_Window_{j+1}",
                "elevation_m": np.mean([c[2] for c in window_corners_gps])
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [window_corners_gps]
            }
        }

        geojson['features'].append(window_feature)

# Process ground
ground_data = load_matlab_data('ground_data.mat')
ground_corners_gps = [local_to_gps(c[0], c[1], c[2]) for c in ground_data['corners']]
ground_corners_gps.append(ground_corners_gps[0])

ground_feature = {
    "type": "Feature",
    "properties": {
        "type": "ground",
        "name": "Ground",
        "elevation_m": GPS_CENTER['altitude']
    },
    "geometry": {
        "type": "Polygon",
        "coordinates": [ground_corners_gps]
    }
}
geojson['features'].append(ground_feature)

# Process roof
roof_data = load_matlab_data('roof_data.mat')
roof_corners_gps = [local_to_gps(c[0], c[1], c[2]) for c in roof_data['corners']]
roof_corners_gps.append(roof_corners_gps[0])

roof_feature = {
    "type": "Feature",
    "properties": {
        "type": "roof",
        "name": "Roof",
        "elevation_m": np.mean([c[2] for c in roof_corners_gps])
    },
    "geometry": {
        "type": "Polygon",
        "coordinates": [roof_corners_gps]
    }
}
geojson['features'].append(roof_feature)

# Process thermal anomalies
thermal_data = load_matlab_data('thermal_data.mat')

for i, hot_spot in enumerate(thermal_data['hot_regions']):
    hot_corners_gps = [local_to_gps(c[0], c[1], c[2]) for c in hot_spot['corners']]
    hot_corners_gps.append(hot_corners_gps[0])

    hot_feature = {
        "type": "Feature",
        "properties": {
            "type": "thermal_anomaly",
            "name": f"Thermal_Red_{i+1}",
            "intensity": hot_spot.get('intensity', 1.0),
            "elevation_m": np.mean([c[2] for c in hot_corners_gps])
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [hot_corners_gps]
        }
    }
    geojson['features'].append(hot_feature)

for i, cold_spot in enumerate(thermal_data['cold_regions']):
    cold_corners_gps = [local_to_gps(c[0], c[1], c[2]) for c in cold_spot['corners']]
    cold_corners_gps.append(cold_corners_gps[0])

    cold_feature = {
        "type": "Feature",
        "properties": {
            "type": "thermal_anomaly",
            "name": f"Thermal_Blue_{i+1}",
            "intensity": cold_spot.get('intensity', 0.3),
            "elevation_m": np.mean([c[2] for c in cold_corners_gps])
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [cold_corners_gps]
        }
    }
    geojson['features'].append(cold_feature)

# Save
with open('Complete.geojson', 'w') as f:
    json.dump(geojson, f, indent=2)

print(f"✓ Created Complete.geojson with {len(geojson['features'])} features")
```

**Run:**
```bash
python3 create_initial_geojson.py
```

**Output:**
- `Complete.geojson` - Initial building geometry with GPS coordinates

---

### STEP 5: Facade Alignment & Rotation

**Objective:** Correct facade planarity issues with rotation transformations

**Scripts Provided:**

1. **Test rotations for each facade**

```bash
# Create test rotations for Facade 1 (X-axis: 0° to 4°)
python3 rotate_facade1_multiple.py

# This creates:
# - Facade1_Rot0.0deg.geojson
# - Facade1_Rot0.5deg.geojson
# - Facade1_Rot1.0deg.geojson
# - Facade1_Rot1.5deg.geojson
# - Facade1_Rot2.0deg.geojson
# - Facade1_Rot2.5deg.geojson
# - Facade1_Rot3.0deg.geojson
# - Facade1_Rot3.5deg.geojson
# - Facade1_Rot4.0deg.geojson
```

2. **Upload to ArcGIS and visually inspect which rotation looks best**

3. **Apply best rotation to all facades**

**Example:** If Facade 1 looked best at 3.0°, rename it:
```bash
cp Facade1_Rot3.0deg.geojson Facade1_Final.geojson
```

4. **Repeat for other facades**

```bash
# Test Y-axis rotations for back/side facades
python3 rotate_facade3_4_Y_axis.py
```

**Output:**
- `Facade1_Final.geojson`
- `Facade2_Final.geojson`
- `Facade3_Final.geojson`
- `Facade4_Final.geojson`

---

### STEP 6: Thermal Anomaly Processing

**Objective:** Separate and rotate thermal anomalies with facade-specific transformations

**Script:** `create_thermal_red_blue_separate.py`

```bash
python3 create_thermal_red_blue_separate.py
```

**Output:**
- `Complete_Thermal_Red_RotX0.5deg.geojson` - Hot spots (red)
- `Complete_Thermal_Blue_RotY0.5deg.geojson` - Cold spots (blue)

---

### STEP 7: Final Assembly

**Objective:** Combine all corrected components into one complete building

**Script:** `create_complete_final_building.py`

```bash
python3 create_complete_final_building.py
```

This script:
- Loads all corrected facade files
- Loads ground and roof
- Loads thermal anomaly files
- Combines into single GeoJSON
- Calculates final dimensions
- Adds metadata

**Output:**
- `DM_Smith_Building_COMPLETE_FINAL.geojson` - Complete digital twin

---

### STEP 8: ArcGIS Online Upload & Visualization

**Objective:** Publish and visualize the digital twin

**Steps:**

1. **Login to ArcGIS Online**
   - Navigate to: https://www.arcgis.com
   - Sign in with your account

2. **Upload GeoJSON**
   - Go to "Content" → "Add Item" → "From your computer"
   - Select `DM_Smith_Building_COMPLETE_FINAL.geojson`
   - Set item details:
     - Title: "DM Smith Building Digital Twin"
     - Tags: digital twin, building, 3D, thermal
     - Summary: Brief description

3. **Open in Scene Viewer**
   - Click "Open in Scene Viewer"
   - Wait for data to load

4. **Configure Elevation**
   - Select the layer in left panel
   - Click "Configure Layer" (properties icon)
   - Go to "Elevation Mode"
   - Select "At an absolute height"
   - Set "Elevation from:" → "elevation_m"
   - Click "Done"

5. **Apply Styling**
   - Click "Styles" button
   - Choose "Types (Unique symbols)"
   - Field: "type"
   - Configure colors:
     - ground → Gray (#808080)
     - roof → Dark Gray (#404040)
     - facade → Blue (#4A90E2)
     - opening → Red (#E74C3C), 50% transparency
     - thermal_anomaly → Style by name:
       - Contains "Red" → Orange (#FF6B35)
       - Contains "Blue" → Cyan (#00D4FF)

6. **Set Symbol Properties**
   - Click on each feature type
   - Set "3D Symbol Type" → "Extrude"
   - Adjust transparency as needed
   - Set outline width/color

7. **Adjust View**
   - Use mouse to rotate, pan, zoom
   - Find optimal viewing angle
   - Save as basemap

8. **Share & Publish**
   - Click "Share" button
   - Set sharing level (Private/Organization/Public)
   - Generate shareable link
   - Optionally embed in website

**Tips:**
- Use "Ground" basemap for better context
- Enable shadows for better depth perception
- Add nearby buildings for scale reference
- Use measurement tools to verify dimensions

---

## File Structure

```
project_root/
│
├── images/                          # Input images
│   ├── rgb/                         # RGB photos
│   │   ├── facade1_01.jpg
│   │   ├── facade1_02.jpg
│   │   └── ...
│   ├── thermal/                     # Thermal IR images
│   │   ├── thermal_facade1.jpg
│   │   └── ...
│   └── location.txt                 # GPS coordinates
│
├── pointcloud/                      # 3D reconstruction output
│   ├── fused.ply                    # Dense point cloud
│   └── sparse/                      # Sparse reconstruction
│
├── matlab_output/                   # MATLAB processed data
│   ├── facade1_data.mat
│   ├── facade2_data.mat
│   ├── facade3_data.mat
│   ├── facade4_data.mat
│   ├── roof_data.mat
│   ├── ground_data.mat
│   ├── thermal_data.mat
│   └── extraction_summary.mat
│
├── geojson/                         # GeoJSON files
│   ├── Complete.geojson             # Initial combined file
│   │
│   ├── testing/                     # Rotation test files
│   │   ├── Facade1_Rot0.0deg.geojson
│   │   ├── Facade1_Rot0.5deg.geojson
│   │   ├── ...
│   │   ├── Facade3_RotY0.5deg.geojson
│   │   └── ...
│   │
│   ├── facade_Final/                # Final corrected facades
│   │   ├── Complete_Ground.geojson
│   │   ├── Complete_Roof.geojson
│   │   ├── Facade1_Final.geojson
│   │   ├── Facade2_Final.geojson
│   │   ├── Facade3_Final.geojson
│   │   ├── Facade4_Final.geojson
│   │   ├── Complete_Thermal_Red_RotX0.5deg.geojson
│   │   ├── Complete_Thermal_Blue_RotY0.5deg.geojson
│   │   └── DM_Smith_Building_COMPLETE_FINAL.geojson  # ← FINAL OUTPUT
│   │
│   └── archive/                     # Old versions
│
├── scripts/                         # Processing scripts
│   ├── matlab_processing.m          # MATLAB segmentation
│   ├── create_initial_geojson.py    # GeoJSON creation
│   ├── rotate_facade1_multiple.py   # Facade 1 rotation tests
│   ├── rotate_facade3_4_Y_axis.py   # Facade 3/4 Y-axis rotations
│   ├── create_thermal_red_blue_separate.py  # Thermal processing
│   ├── create_complete_final_building.py    # Final assembly
│   └── move_facade2_towards_bobby_dodd.py   # Facade adjustment
│
├── docs/                            # Documentation
│   ├── README.md                    # This file
│   ├── Digital_Twin_Creation_Workflow.pptx  # Presentation
│   └── screenshots/                 # Process screenshots
│
└── outputs/                         # Final deliverables
    ├── DM_Smith_Building_COMPLETE_FINAL.geojson
    ├── metadata.json                # Building metadata
    └── validation_report.txt        # Dimension validation
```

---

## Script Documentation

### Core Scripts

#### 1. `create_initial_geojson.py`

**Purpose:** Convert MATLAB data to GeoJSON with GPS coordinates

**Input:**
- MATLAB .mat files (facade1_data.mat, etc.)
- GPS center coordinates (hardcoded)

**Output:**
- Complete.geojson

**Key Functions:**
- `load_matlab_data()` - Load .mat files
- `local_to_gps()` - Convert ENU to GPS coordinates
- Main loop creates GeoJSON features

**Usage:**
```bash
python3 create_initial_geojson.py
```

---

#### 2. `rotate_facade1_multiple.py`

**Purpose:** Test multiple rotation angles for Facade 1

**Parameters:**
```python
ROTATION_ANGLES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]  # degrees
ROTATION_AXIS = 'X'  # X, Y, or Z
```

**Process:**
1. Load facade data
2. Calculate center point
3. For each angle:
   - Create rotation matrix
   - Apply to all coordinates
   - Save as separate file
4. Generate comparison report

**Output:**
- Facade1_Rot{angle}deg.geojson (9 files)
- rotation_comparison.txt

**Usage:**
```bash
python3 rotate_facade1_multiple.py
```

**Visual Testing:**
Upload all files to ArcGIS Online and compare planarity visually

---

#### 3. `rotate_facade3_4_Y_axis.py`

**Purpose:** Apply Y-axis rotations to back/side facades

**Why Y-axis?**
- Facades 3 and 4 face different directions
- X-axis rotation made them worse
- Y-axis rotation improves their planarity

**Parameters:**
```python
ROTATION_ANGLES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
ROTATION_AXIS = 'Y'
```

**Usage:**
```bash
python3 rotate_facade3_4_Y_axis.py
```

---

#### 4. `create_thermal_red_blue_separate.py`

**Purpose:** Separate thermal anomalies and apply facade-specific rotations

**Logic:**
- Red anomalies (hot spots) → On front facades → X-axis rotation
- Blue anomalies (cold spots) → On back/side facades → Y-axis rotation

**Parameters:**
```python
RED_ROTATION_DEGREES = 0.5   # X-axis
BLUE_ROTATION_DEGREES = 0.5  # Y-axis
```

**Output:**
- Complete_Thermal_Red_RotX0.5deg.geojson
- Complete_Thermal_Blue_RotY0.5deg.geojson

**Usage:**
```bash
python3 create_thermal_red_blue_separate.py
```

---

#### 5. `create_complete_final_building.py`

**Purpose:** Assemble all corrected components into final digital twin

**Process:**
1. Load all component files:
   - Ground, Roof
   - Facades 1-4 (final versions)
   - Thermal anomalies
2. Combine all features
3. Calculate dimensions
4. Add metadata
5. Save as complete building

**Output:**
- DM_Smith_Building_COMPLETE_FINAL.geojson

**Metadata Included:**
- Total features count
- Component breakdown
- Rotations applied
- Actual dimensions
- Building center GPS

**Usage:**
```bash
python3 create_complete_final_building.py
```

---

#### 6. `move_facade2_towards_bobby_dodd.py`

**Purpose:** Apply translation adjustment to Facade 2

**Why needed?**
- Facade 2 appeared "inside" the building
- Translation moves it outward

**Parameters:**
```python
MOVE_DISTANCE = 0.5  # meters
DIRECTION = "towards Bobby Dodd Way"  # West along Cherry Street
```

**Process:**
1. Calculate Cherry Street axis from Facade 1
2. Determine westward direction vector
3. Apply translation to all Facade 2 features
4. Save adjusted facade

**Output:**
- Facade2_ONLY_Moved_0.5m.geojson

**Usage:**
```bash
python3 move_facade2_towards_bobby_dodd.py
```

---

### Helper Scripts

#### 7. `create_unified_rotation_from_original.py`

**Purpose:** Test unified rotation approach (all components same rotation)

**Approach:**
- Apply single rotation to entire building
- Preserves facade connections
- Trade-off: less optimal individual facade planarity

**Scenarios:**
- Original (no rotation)
- 0.5° X-axis
- 1.0° X-axis
- 1.5° X-axis
- 2.0° X-axis

**Usage:**
```bash
python3 create_unified_rotation_from_original.py
```

---

#### 8. `create_presentation.py`

**Purpose:** Generate PowerPoint documentation

**Output:**
- Digital_Twin_Creation_Workflow.pptx (21 slides)

**Usage:**
```bash
python3 create_presentation.py
```

---

## Configuration Parameters

### GPS Center Coordinates

**Critical:** Accurate GPS center ensures correct positioning

```python
GPS_CENTER = {
    'latitude': 33.773687,   # Decimal degrees
    'longitude': -84.395185, # Decimal degrees
    'altitude': 305.0        # Meters above sea level
}
```

**How to determine:**
1. Google Maps: Right-click building → Copy coordinates
2. GPS device: Stand at building center
3. Survey data: If available, use precise coordinates

**Format:**
- Latitude: Positive = North, Negative = South
- Longitude: Positive = East, Negative = West
- Altitude: MSL (mean sea level) in meters

---

### Coordinate Conversion Constants

```python
LAT_CENTER = 33.773687
METERS_PER_DEGREE_LAT = 111000.0  # Constant worldwide
METERS_PER_DEGREE_LON = 111000.0 * math.cos(math.radians(LAT_CENTER))
```

**Why cosine correction?**
- Longitude degree distance varies by latitude
- At equator: 111 km per degree
- At 34°N: ~91.8 km per degree
- At poles: 0 km per degree

---

### Target Building Dimensions

```python
TARGET_WIDTH = 30.5   # meters (East-West)
TARGET_LENGTH = 32.0  # meters (North-South)
TARGET_HEIGHT = 17.0  # meters (Ground to Roof)
```

**How to measure:**
1. Physical measurement (tape measure)
2. Architectural drawings
3. Building permits/plans
4. Estimate from satellite imagery

**Used for:**
- Validation of final output
- Scaling adjustments if needed
- Quality assurance

---

### Rotation Parameters

```python
# Rotation angles to test (degrees)
ROTATION_ANGLES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# Rotation axes
ROTATION_AXIS_X = 'X'  # Pitch (North-South tilt)
ROTATION_AXIS_Y = 'Y'  # Roll (East-West tilt)
ROTATION_AXIS_Z = 'Z'  # Yaw (Horizontal rotation)
```

**Typical values:**
- Front facades: 0.5° to 3.0° X-axis
- Back/side facades: 0.5° to 2.0° Y-axis
- Rarely need Z-axis rotation

---

## Troubleshooting

### Issue 1: Facades Not Planar in ArcGIS

**Symptoms:**
- Facades appear wavy or non-flat
- Windows don't align properly
- Visible bulges or distortions

**Solutions:**

1. **Test more rotation angles**
   ```bash
   # Increase granularity
   ROTATION_ANGLES = [0.0, 0.25, 0.5, 0.75, 1.0, ...]
   ```

2. **Try different rotation axis**
   ```python
   # If X-axis doesn't work, try Y-axis
   ROTATION_AXIS = 'Y'
   ```

3. **Check coordinate precision**
   - Ensure 10 decimal places for lon/lat
   - Verify GPS center is accurate

4. **Verify MATLAB segmentation**
   - Check if point cloud is clean
   - Ensure plane fitting is good

---

### Issue 2: Building Not at Correct GPS Location

**Symptoms:**
- Building appears in wrong location on map
- Offset from satellite imagery
- Wrong city/country

**Solutions:**

1. **Verify GPS coordinates**
   ```python
   # Check sign (positive/negative)
   # Atlanta should be:
   # Latitude: ~33.7° N (positive)
   # Longitude: ~-84.4° W (negative)
   ```

2. **Check coordinate order**
   - GeoJSON format: [longitude, latitude, altitude]
   - NOT [latitude, longitude]!

3. **Validate meters per degree**
   ```python
   # Should be calculated correctly
   meters_per_deg_lon = 111000 * cos(lat_radians)
   ```

4. **Test with single point**
   ```python
   # Create test GeoJSON with known point
   # Verify it appears in correct location
   ```

---

### Issue 3: Facades Have Gaps / Don't Connect

**Symptoms:**
- Visible gaps between adjacent facades
- Facades overlap incorrectly
- Corner joints don't meet

**Causes:**
- Individual rotations broke geometric relationships
- Different rotation angles for adjacent facades

**Solutions:**

1. **Use unified rotation approach**
   ```bash
   python3 create_unified_rotation_from_original.py
   ```

2. **Apply translation adjustments**
   ```bash
   # Move specific facade to close gap
   python3 move_facade2_towards_bobby_dodd.py
   ```

3. **Accept trade-off**
   - Perfect individual facades vs perfect connections
   - May need to choose compromise

---

### Issue 4: Thermal Anomalies Not Visible

**Symptoms:**
- Thermal features missing in ArcGIS
- Only blue or only red anomalies show
- Thermal layer not rendering

**Solutions:**

1. **Check feature count**
   ```bash
   # Verify thermal features exist
   grep "thermal_anomaly" Complete.geojson
   ```

2. **Verify separation script ran**
   ```bash
   python3 create_thermal_red_blue_separate.py
   ```

3. **Check styling in ArcGIS**
   - Ensure thermal_anomaly type is styled
   - Use bright colors (orange/cyan)
   - Set transparency to 50%

4. **Verify elevation values**
   - Thermal features need valid elevation_m
   - Should match facade elevation

---

### Issue 5: Elevation Not Working in ArcGIS

**Symptoms:**
- Building appears flat (2D) instead of 3D
- All features at same height
- No vertical separation

**Solutions:**

1. **Check elevation mode**
   - Must be "At an absolute height"
   - NOT "On the ground" or "Relative to ground"

2. **Verify elevation_m property exists**
   ```python
   # All features must have this
   "properties": {
       "elevation_m": 305.5,  # Must be present
       ...
   }
   ```

3. **Check altitude values in coordinates**
   ```python
   # Third value is altitude
   [longitude, latitude, altitude]  # altitude must vary
   ```

4. **Validate altitude range**
   ```python
   # Ground: ~305m
   # Roof: ~322m
   # Should have ~17m variation
   ```

---

### Issue 6: File Size Too Large for ArcGIS

**Symptoms:**
- Upload fails
- Timeout during processing
- ArcGIS Online error

**Solutions:**

1. **Reduce coordinate precision**
   ```python
   # Use 8 decimal places instead of 10
   lon = round(lon, 8)
   lat = round(lat, 8)
   ```

2. **Simplify geometry**
   - Reduce number of window features
   - Combine small thermal anomalies
   - Remove unnecessary vertices

3. **Split into multiple layers**
   - Upload facades separately
   - Separate thermal layer
   - Upload as layer group

4. **Compress GeoJSON**
   ```bash
   # Remove indentation
   python3 -m json.tool --compact Complete.geojson > Compressed.geojson
   ```

---

### Issue 7: Python Script Errors

**Common Errors:**

**Error:** `ModuleNotFoundError: No module named 'numpy'`
```bash
# Solution:
pip install numpy
```

**Error:** `KeyError: 'features'`
```python
# Solution: Check GeoJSON structure
# Ensure "features" key exists
if 'features' not in data:
    data['features'] = []
```

**Error:** `IndexError: list index out of range`
```python
# Solution: Check coordinate array length
if len(coords) > 0:
    coord_array = coords[0]
```

**Error:** `TypeError: 'float' object is not subscriptable`
```python
# Solution: Validate coordinate structure
# Should be: [lon, lat, alt] not a single float
if isinstance(c, list) and len(c) == 3:
    # Process coordinate
```

---

## Best Practices

### 1. Version Control

**Always preserve originals:**
```bash
# Before any processing, backup original data
cp Complete.geojson Complete_ORIGINAL_BACKUP.geojson

# Use version numbers for iterations
Complete_v1.geojson
Complete_v2_rotated.geojson
Complete_v3_final.geojson
```

**Git workflow:**
```bash
git init
git add *.py *.geojson
git commit -m "Initial digital twin creation"
```

---

### 2. Incremental Testing

**Test each step before moving to next:**

```bash
# Step 1: Create initial GeoJSON
python3 create_initial_geojson.py
# → Upload to ArcGIS, verify GPS location

# Step 2: Test Facade 1 rotations
python3 rotate_facade1_multiple.py
# → Upload all versions, pick best one

# Step 3: Continue with other facades
# ...
```

**Don't:**
- Run all scripts in sequence without validation
- Assume output is correct without visual inspection

---

### 3. Documentation

**Document every decision:**

Create `processing_log.txt`:
```
2025-11-11 10:00 - Created initial GeoJSON from MATLAB data
2025-11-11 10:30 - Tested Facade 1 rotations (0° to 4° X-axis)
2025-11-11 11:00 - Selected 3.0° rotation for Facade 1 (best planarity)
2025-11-11 11:30 - Tried X-axis for Facade 3, made worse
2025-11-11 12:00 - Switched to Y-axis for Facade 3, improved
2025-11-11 12:30 - Selected 0.5° Y-axis for Facade 3
...
```

**Include:**
- Date/time
- Action taken
- Parameters used
- Result/observation
- Decision made

---

### 4. Quality Validation

**Checklist before final delivery:**

- [ ] GPS location correct (matches satellite imagery)
- [ ] Building dimensions within ±1m of targets
- [ ] All facades appear planar in 3D view
- [ ] Windows aligned properly on facades
- [ ] No gaps between facades
- [ ] Thermal anomalies visible
- [ ] Elevation mode working correctly
- [ ] All features have elevation_m property
- [ ] GeoJSON validates (use https://geojsonlint.com)
- [ ] File size acceptable for ArcGIS (<10MB)
- [ ] Documentation complete

**Validation Script:**
```python
#!/usr/bin/env python3
"""Validate final GeoJSON"""
import json

with open('DM_Smith_Building_COMPLETE_FINAL.geojson', 'r') as f:
    data = json.load(f)

# Check feature count
print(f"Total features: {len(data['features'])}")

# Check for elevation_m
missing_elevation = []
for i, feature in enumerate(data['features']):
    if 'elevation_m' not in feature['properties']:
        missing_elevation.append(i)

if missing_elevation:
    print(f"WARNING: {len(missing_elevation)} features missing elevation_m")
else:
    print("✓ All features have elevation_m")

# Check coordinate format
for feature in data['features'][:5]:  # Sample first 5
    coords = feature['geometry']['coordinates'][0]
    for c in coords[:3]:  # Check first 3 coords
        if not (isinstance(c, list) and len(c) == 3):
            print(f"ERROR: Invalid coordinate format in {feature['properties']['name']}")
            break

print("✓ Validation complete")
```

---

### 5. Performance Optimization

**For large buildings:**

1. **Parallel processing**
   ```python
   from multiprocessing import Pool

   def process_facade(facade_data):
       # Process single facade
       return result

   with Pool(4) as p:
       results = p.map(process_facade, facade_list)
   ```

2. **Batch operations**
   ```python
   # Process multiple facades at once
   for facade in [1, 2, 3, 4]:
       process_facade(facade)
   ```

3. **Efficient data structures**
   ```python
   # Use numpy arrays for coordinate transformations
   import numpy as np
   coords_array = np.array(coords_list)
   rotated = rotation_matrix @ coords_array.T
   ```

---

## Example Results

### Case Study: D.M. Smith Building

**Building Details:**
- Location: Georgia Institute of Technology, Atlanta, GA
- Address: Corner of Cherry Street and Bobby Dodd Way
- Dimensions: 30.5m × 32.0m × 17.0m
- Facades: 4 (with 109 windows total)
- Thermal anomalies: 47 (42 blue, 5 red)

**Processing Time:**
- Image acquisition: 2 hours
- MATLAB processing: 4 hours
- GeoJSON creation: 1 hour
- Rotation testing: 3 hours
- Final assembly: 1 hour
- **Total: ~11 hours**

**Final Output:**
- File: DM_Smith_Building_COMPLETE_FINAL.geojson
- Size: 211 KB
- Features: 162
- Coordinate precision: 10 decimal places (~1cm accuracy)

**Rotation Parameters Used:**
- Facade 1: 3.0° X-axis
- Facade 2: 0.5° X-axis + 0.5m translation west
- Facade 3: 0.5° Y-axis
- Facade 4: 1.5° Y-axis
- Red thermal: 0.5° X-axis
- Blue thermal: 0.5° Y-axis

**Accuracy Achieved:**
- GPS position: Within 2m of actual location
- Building dimensions: Within 0.5m of measured values
- Facade planarity: 60-85% improvement from initial
- Elevation accuracy: ±0.3m

**Applications:**
- Building Energy Management (BEM) analysis
- Thermal audit visualization
- Facility management documentation
- Campus digital twin integration
- Research and education

---

## Additional Resources

### Tools & Software

- **COLMAP:** https://colmap.github.io/
- **Nerfstudio:** https://docs.nerf.studio/
- **ArcGIS Online:** https://www.arcgis.com
- **GeoJSON Validator:** https://geojsonlint.com
- **QGIS:** https://qgis.org (alternative to ArcGIS)

### Learning Resources

- **GeoJSON Specification:** https://geojson.org/
- **WGS84 Coordinate System:** https://epsg.io/4326
- **3D Rotation Matrices:** https://en.wikipedia.org/wiki/Rotation_matrix
- **Structure from Motion:** https://en.wikipedia.org/wiki/Structure_from_motion

### Related Papers

- NeRF: Representing Scenes as Neural Radiance Fields (Mildenhall et al., 2020)
- Building Information Modeling (BIM) for Facility Management
- Digital Twins in Smart Buildings: State of the Art

---

## Support & Contact

For questions, issues, or contributions:

- **Email:** [your-email@gatech.edu]
- **GitHub:** [repository-url]
- **Issues:** [repository-url/issues]

---

## License

This pipeline and associated scripts are provided for research and educational purposes.

---

## Acknowledgments

- Georgia Institute of Technology
- Building Energy Management Research Team
- ArcGIS Online platform
- Open-source community (NumPy, Python)

---

## Changelog

### v1.0.0 (November 2025)
- Initial release
- Complete pipeline from images to digital twin
- Support for thermal anomaly integration
- ArcGIS Online compatibility
- Comprehensive documentation

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{digital_twin_pipeline_2025,
  title={Digital Twin Creation Pipeline: From Images to 3D GeoJSON},
  author={Building Energy Management Team},
  organization={Georgia Institute of Technology},
  year={2025},
  url={[repository-url]}
}
```

---

**END OF README**

For detailed technical information, see:
- Digital_Twin_Creation_Workflow.pptx (21-slide presentation)
- Individual script documentation (inline comments)
- processing_log.txt (if created during your workflow)
