# GLRBM-Mocap

Motion capture fitting using marker data from CMU and BMLrub datasets.

## Prerequisites

Before running the code, you need to prepare the data:

1. Download the SMPL-H files for CMU and BMLrub datasets from the [AMASS website](https://amass.is.tue.mpg.de/).
2. Extract the downloaded data and place them in the `data/` folder in the project root directory.
3. Uncomment the data processing code in `data.py` and run it to convert the data to LMDB format:

```bash
python data.py
```

This will generate the processed datasets in LMDB format required for training.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Training

To train a model, use the following command:

```bash
python mocap_fitter.py --base_model sequence --model_type transformer --train_mode train --marker_type rbm --epochs 1000 --lr_decay --use_geodesic_loss --use_rela_x
```

The trained model will be saved in the `results/` folder in the project root directory.

### Training Parameters

- `--base_model`: Model architecture type (`sequence`, `frame`, or `continuity`)
- `--model_type`: Model type (`rnn`, `lstm`, `gru`, `cnn`, or `transformer`)
- `--train_mode`: Training mode (`train` or `test`)
- `--marker_type`: Marker type (`moshpp`, `rbm`, `rbm_a`, `rbm_b`, `rbm_c`, `rbm_d`, `rbm_e`, or `rbm_f`)
- `--epochs`: Number of training epochs
- `--lr_decay`: Enable learning rate decay
- `--use_geodesic_loss`: Use geodesic loss for pose estimation
- `--use_rela_x`: Use relative coordinates

## Testing

To test a trained model, run:

```bash
python mocap_fitter.py --model_path results/your_result --train_mode test
```

Replace `results/your_result` with the path to your trained model directory. The script will automatically load the configuration from the `config.json` file in the model directory.

## Visualization

To visualize the results during testing, set the `vis` parameter to `True` in the `test` function in `mocap_fitter.py`:

```python
test(
    ...
    vis=True,
    ...
)
```

## Project Structure

- `mocap_fitter.py`: Main training and testing script
- `data.py`: Data processing and dataset loading
- `models.py`: Model architectures
- `metric.py`: Evaluation metrics
- `smpl.py`: SMPL model implementation
- `utils.py`: Utility functions including visualization

## License

[Add your license information here]

