# Point Cloud Completion - Python Implementation

This is a Python implementation of the MATLAB point cloud completion system for buildings. The code has been converted to maintain identical processing logic and output results.

## Features

- **Point Cloud Downsampling**: Reduces point density for efficient processing
- **Noise Removal**: Filters out noisy points using k-nearest neighbor analysis
- **Orientation Adjustment**: Uses BBVM (Bounding Box Volume Minimization) method
- **Surface Completion**: Completes missing surfaces in 6 directions (x1, x2, y1, y2, z1, z2)
- **Iterative Refinement**: Multi-layer completion process
- **Result Filtering**: Removes redundant points and refines surfaces

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
python main.py
```

This will process the default input file `B7.txt` and generate:
- `data_new.txt`: Intermediate result after orientation adjustment
- `data_final.txt`: Final completed point cloud

### Test Mode

```python
python main.py test
```

This will run a test with sample data to verify the system is working correctly.

### Custom Input File

Edit the `input_file` variable in `main.py`:

```python
input_file = 'your_point_cloud.txt'  # Replace with your filename
```

## Input Format

The input file should be a text file with point cloud data:
- Each line represents one point
- Columns: X Y Z [R G B] (RGB colors are optional)
- Separated by spaces or tabs

Example:
```
1.234567 2.345678 3.456789
1.234568 2.345679 3.456790
...
```

## Output Format

The output files contain the processed point cloud data:
- `data_new.txt`: Points after orientation adjustment
- `data_final.txt`: Final completed point cloud

## Processing Pipeline

1. **Data Loading**: Read point cloud from text file
2. **Downsampling**: Reduce point density (default: 20% of original)
3. **Noise Removal**: Filter points using k-NN distance analysis
4. **Orientation Adjustment**: Rotate to optimal building orientation
5. **Surface Completion**: Fill missing surfaces in 6 directions
6. **Iterative Refinement**: Multi-layer completion process
7. **Coordinate Transformation**: Transform back to original coordinates
8. **Final Filtering**: Remove redundant points
9. **Result Saving**: Save final completed point cloud

## Key Functions

### PointCloudProcessor
Main class for point cloud processing operations:
- `read_point_cloud()`: Load data from file
- `pointcloud_downsample()`: Reduce point density
- `datas_quzao()`: Remove noise
- `tiaozheng_zitai3()`: Adjust orientation
- `save_data()`: Save results to file

### SurfaceCompletion
Class for surface completion algorithms:
- `quedingbianjie()`: Determine boundaries
- `down_buliao()`: Downward surface completion
- `fantan()`: Surface interpolation
- `wanggehua()`: Create polygons
- `refine_buliao3()`: Refine results

### BuliaoMoni
Main surface completion function:
- `buliaomoni()`: Complete missing surfaces

## Parameters

Key parameters that can be adjusted:

- **Downsampling ratio**: `0.2` (20% of original points)
- **Noise threshold**: `1.0` (k-NN distance threshold)
- **Completion distance**: `5` (surface completion distance)
- **Refinement flag**: `0` or `1` (enable/disable refinement)

## Comparison with MATLAB

This Python implementation maintains identical logic to the original MATLAB code:

- Same processing pipeline
- Same mathematical operations
- Same parameter values
- Same output format

The main differences are:
- Python syntax and libraries
- Error handling improvements
- Better code organization
- Enhanced logging

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
2. **File Not Found**: Check that input file exists and path is correct
3. **Memory Issues**: For large point clouds, reduce downsampling ratio
4. **Empty Results**: Check input data format and quality

### Performance Tips

- Use downsampling for large point clouds (>100k points)
- Adjust noise removal threshold based on data quality
- Enable refinement only for high-quality results

## License

This code is provided as-is for research and educational purposes.

## Contact

For questions or issues, please refer to the original MATLAB implementation documentation.
