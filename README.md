# Machine Learning Term Project

This repository contains the code for our term project in the **Machine Learning** course at PKU, which focuses on the **reproduction and extension** of key methods from two major works in the field of lattice quantization:

1. *"Optimization and Identification of Lattice Quantizers"* (Erik Agrell, Daniel Pook-Kolb, & Bruce Allen, 2024)
2. *"On the Best Lattice Quantizers"* (Erik Agrell & Bruce Allen, 2023)

Our work primarily involves the replication and improvement of methods described in these two papers. We have implemented and optimized algorithms from the first paper, which includes both the optimization of lattice generators and the visualization of theta images. The second paper's methods are used for updating the lattice quantization table, constructing high-dimensional generator matrices from low-dimensional ones.

### Research Papers Reproduced:
1. **"Optimization and Identification of Lattice Quantizers"**  
   Erik Agrell, Daniel Pook-Kolb, & Bruce Allen (2024).  
   [Link to paper](https://arxiv.org/abs/2401.01799)

2. **"On the Best Lattice Quantizers"**  
   Erik Agrell & Bruce Allen (2023).  
   [Link to paper](http://dx.doi.org/10.1109/TIT.2023.3291313)

## Project Overview

This project consists of several Python scripts, each replicating different aspects of the lattice-based quantization techniques described in the referenced papers.

### Main Contributions:
1. **Reproducing and improving the optimization and theta image visualization methods** from *"Optimization and Identification of Lattice Quantizers"* (Agrell et al., 2024). This involves numerical optimization algorithms for lattice construction, which have been further refined and extended in this work.
2. **Updating the lattice quantization table** in *"On the Best Lattice Quantizers"* (Agrell & Allen, 2023). This method constructs high-dimensional lattices from low-dimensional generator matrices and shows improvements over the current state-of-the-art (SOTA) in certain dimensions.
   
   The low-dimensional NSM data used in this work come from various sources, including the referenced works and our own experiments, and demonstrate an improvement in the lattice quantization table for some dimensions.

## File Structure

- **`lattice_constructor.py`**: Implements the optimization algorithm for lattice construction based on *"Optimization and Identification of Lattice Quantizers"* (Agrell et al., 2024). This script generates the lattice structure and saves the generator matrices.
- **`visualize_theta.py`**: Visualizes the theta image from the lattice generators, offering a new way of characterizing lattices as proposed in *"Optimization and Identification of Lattice Quantizers"*.
- **`nsm.py`**: Calculates the Normalized Squared Metric (NSM) using Monte Carlo simulations, as described in the *"Optimization and Identification of Lattice Quantizers"*. The NSM helps quantify the quality of the lattice and is an essential part of the optimization process.
- **`product_lattice.py`**: Implements the method described in *"On the Best Lattice Quantizers"* (Agrell & Allen, 2023) to update the lattice quantization table using low-dimensional generator matrices to construct high-dimensional lattices. This method improves upon the current SOTA for certain dimensions.

## Requirements

To run the code, you need Python along with the following libraries:

- numpy
- matplotlib

## Usage

### 1. Lattice Construction and Optimization

This script implements the optimization algorithm for lattice construction based on the method described in *"Optimization and Identification of Lattice Quantizers"*. To run the `lattice_constructor.py` script:

```bash
python lattice_constructor.py
```

You can customize the following parameters:

- `dim`: The dimension of the lattice (default: 2).
- `mode`: The optimization mode for lattice construction ('fast', 'medium', 'slow', or 'diy').
- `mu`: Initial step size (default: 0.005).
- `nu`: Ratio between initial and final step size (default: 200).
- `step`: Number of iteration steps (default: 1000000).
- `Tr`: Reduction interval (default: 100).

For example, to customize the parameters:

```bash
python lattice_constructor.py --dim 3 --mode fast
```

Or to use the `diy` mode with custom values:

```bash
python lattice_constructor.py --dim 3 --mode diy --mu 0.01 --nu 100 --step 2000000 --tr 50
```

The generator matrices will be saved in a file named `record_{DIM}_{MODE}.npy`.

### 2. Theta Image Visualization

Once the lattice data is generated, you can visualize the corresponding theta image using the `visualize_theta.py` script. This method visualizes the lattice structure from a different viewpoint, as suggested in *"Optimization and Identification of Lattice Quantizers"*.

```bash
python visualize_theta.py --dim DIM --mode MODE
```

This will load the lattice data from `record_{DIM}_{MODE}.npy` and generate the corresponding theta image.

### 3. NSM Calculation with Monte Carlo

The `nsm.py` script calculates the Normalized Squared Metric (NSM) using Monte Carlo simulations, which is an essential part of the optimization process in *"Optimization and Identification of Lattice Quantizers"*. To run this script:

```bash
python nsm.py
```

You can customize the following parameters:

- `dim`: The dimension of the lattice (default: 2).
- `mode`: The optimization mode for lattice construction ('fast', 'medium', 'slow', or 'diy').
- `step`: The number of Monte Carlo simulation steps (default: 1000000).

For example, to run with custom parameters:

```bash
python nsm.py --dim 3 --mode fast --step 500000
```

### 4. Product Lattice NSM Calculation

The `product_lattice.py` script calculates the NSM for product lattices, as described in *"On the Best Lattice Quantizers"*. This method constructs high-dimensional lattices from low-dimensional generator matrices, improving the lattice quantization table in certain dimensions. To run the script:

```bash
python product_lattice.py
```

Currently, this script does not have customizable parameters, but you can modify the code as needed to suit your experiments.

## Optional Parameters

- `dim`: The dimension of the lattice (default is 2).
- `mode`: The optimization mode for lattice construction ('fast', 'medium', 'slow', or 'diy').
- `mu`: Initial step size (default is 0.005).
- `nu`: Ratio between initial and final step size (default is 200).
- `step`: Number of iteration steps (default is 1000000).
- `Tr`: Reduction interval (default is 100).

## Group Members

- Yangdi Yue
- Hongxi Song
- Zhining Zhang
- Yihan Geng

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.