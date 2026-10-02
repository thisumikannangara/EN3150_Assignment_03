# Optimizer Selection and Custom Model Training

## 1. Experimental Setup

The experiments were conducted using the EuroSAT RGB dataset. The dataset consists of RGB satellite images with a spatial resolution of 64 × 64 pixels and 10 land-use classes.

A fixed stratified dataset split was used throughout the experiments:

* **Training set:** 70%
* **Validation set:** 15%
* **Test set:** 15%
* **Random seed:** 42
* **Batch size:** 64
* **Loss function:** Cross-Entropy Loss
* **Device:** CPU
* **PyTorch version:** 2.14.1+cpu
* **Number of epochs:** 20

The same dataset split, preprocessing, and training settings were maintained across the experiments to ensure a fair comparison between the different optimizers and custom CNN models.
## 2. Optimizer Configurations

Three optimization methods were considered for training the custom CNN models. The learning rates and momentum values were selected according to the experimental configuration.

| Optimizer | Learning rate | Momentum | Epochs in recorded run |
|---|---:|---:|---:|
| SGD | 0.01 | – | 20 |
| SGD with Momentum | 0.01 | 0.9 | 20 |
| Adam | 0.001 | – | 20 |

The requested optimizer-comparison duration was 5 epochs, but the available experiment script, result CSV and saved curves document 20-epoch runs. The table and conclusions below therefore report the recorded 20-epoch experiments rather than presenting unverified 5-epoch results.

## 3. Optimizer Results

The requested file `outputs/member3/metrics/optimizer_comparison.csv` and consolidated `optimizer_validation_loss.png` and `optimizer_validation_accuracy.png` figures are not present in the workspace. The available source is `results/member3/model_optimizer_comparison.csv`. It contains test metrics and average epoch times, but not validation-accuracy histories or best validation losses. Therefore, unavailable validation metrics are marked as not recorded; they are not inferred from plots.

The following comparison uses Model A, which was the architecture used to select the optimizer. All metrics and times below are copied from the available comparison CSV and rounded for readability.

| Optimizer | Best validation accuracy | Best validation loss | Test accuracy | Macro precision | Macro recall | Average epoch time (s) | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| SGD | Not recorded | Not recorded | 72.74% | 71.63% | 71.75% | 101.03 | No |
| SGD with Momentum | Not recorded | Not recorded | 89.41% | 89.27% | 89.38% | 111.91 | **Yes** |
| Adam | Not recorded | Not recorded | 89.16% | 89.48% | 88.65% | 141.21 | No |

SGD with Momentum was selected because, in the recorded Model A runs, it had the highest test accuracy (89.41%) and macro recall (89.38%). Its accuracy was 0.25 percentage points higher than Adam's and 16.67 points higher than plain SGD's. Adam had marginally higher macro precision (89.48% versus 89.27%), but lower test accuracy and macro recall, and its average epoch time was longer. This selection is based on the recorded results, not an assumption that Adam is best.

Available Model A training/validation-loss curves (each shows both loss series):

| Optimizer | Loss curve |
|---|---|
| SGD | ![Model A SGD training and validation loss](../../results/member3/ModelA_SGD_loss_curve.png) |
| SGD with Momentum | ![Model A SGD with Momentum training and validation loss](../../results/member3/ModelA_SGD_Momentum_loss_curve.png) |
| Adam | ![Model A Adam training and validation loss](../../results/member3/ModelA_Adam_loss_curve.png) |

A validation-accuracy curve was not saved, so validation accuracy and its best epoch cannot be reported from the current artifacts.

## 4. Effect of Momentum

SGD updates parameters using the current gradient alone. Because gradients can vary between mini-batches, plain SGD may change direction repeatedly and oscillate. Momentum accumulates information from previous update directions, giving updates a component of their recent history. This tends to smooth progress and can help optimization continue through shallow or noisy regions.

In the saved Model A curves, plain SGD's validation loss has several pronounced upward spikes (including around epochs 10, 14 and 17). The momentum curve decreases more steadily, with smaller temporary reversals, and finishes at a visibly lower validation loss. This is a qualitative reading of the plotted curves; exact minimum validation-loss values were not saved. The measured test accuracy also improved, from 72.74% with SGD to 89.41% with momentum. A validation-accuracy curve or numerical validation-accuracy history is absent, so a validation-accuracy improvement cannot be quantified from these artifacts.

## 5. Model A Training and Evaluation

Model A was trained for 20 epochs using SGD with Momentum (learning rate 0.01, momentum 0.9), the selected optimizer. The loss-curve figure contains the training-loss and validation-loss series. The available test metrics and mean epoch time are:

| Measure | Model A result |
|---|---:|
| Test accuracy | 89.41% |
| Macro precision | 89.27% |
| Macro recall | 89.38% |
| Average training time per epoch | 111.91 s |

![Model A training and validation loss over 20 epochs](../../results/member3/ModelA_SGD_Momentum_loss_curve.png)

The test-set confusion matrix is shown below.

![Model A SGD with Momentum confusion matrix](../../results/member3/ModelA_SGD_Momentum_confusion_matrix.png)

## 6. Model B Training and Evaluation

For a direct comparison with Model A, Model B was also trained for 20 epochs using SGD with Momentum (learning rate 0.01, momentum 0.9). Its saved loss curve contains the training-loss and validation-loss series. The recorded metrics are:

| Measure | Model B result |
|---|---:|
| Test accuracy | 62.84% |
| Macro precision | 59.92% |
| Macro recall | 61.01% |
| Average training time per epoch | 190.52 s |

![Model B training and validation loss over 20 epochs](../../results/member3/ModelB_SGD_Momentum_loss_curve.png)

The test-set confusion matrix is shown below.

![Model B SGD with Momentum confusion matrix](../../results/member3/ModelB_SGD_Momentum_confusion_matrix.png)

## 7. Model A versus Model B

The available experiment comparison CSV records both models with SGD with Momentum. Parameter counts and actual serialized model sizes are recorded in the architecture report. The serialized sizes are used here, rather than theoretical FP32 weight sizes.

| Measurement | Model A | Model B |
|---|---:|---:|
| Trainable parameters | 102,154 | 11,080 |
| Actual serialized model size | 402.82 KB | 49.67 KB |
| Average time per epoch | 111.91 s | 190.52 s |
| Test accuracy | 89.41% | 62.84% |
| Macro precision | 89.27% | 59.92% |
| Macro recall | 89.38% | 61.01% |

## 8. Discussion

- **Accuracy:** Model A achieved the higher test accuracy: 89.41%, compared with 62.84% for Model B.
- **Training speed:** Model A trained faster in these CPU runs, averaging 111.91 seconds per epoch versus 190.52 seconds for Model B. Thus, the lightweight architecture's lower parameter count did not translate into faster training in this experiment.
- **Storage:** Model B used less storage: 49.67 KB serialized versus 402.82 KB for Model A, about 87.67% smaller. It also has about 89.15% fewer trainable parameters.
- **Accuracy trade-off:** Model B lost 26.57 percentage points of test accuracy relative to Model A when both used the selected optimizer.
- **Edge-device suitability:** Model B is more suitable where model storage or parameter memory is the main constraint. However, the measured CPU epoch time was longer and accuracy was substantially lower, so these results alone do not establish that it is preferable for an edge deployment. Inference latency, memory use and energy on the target device were not recorded.
- **Overall trade-off:** The storage reduction may be worthwhile for a severely memory-constrained deployment, but the current evidence does not show a computation-time benefit and does show a sizable accuracy cost. For an application prioritizing classification accuracy, Model A is preferable; for a strict storage budget, Model B may be considered only if its accuracy is acceptable for the task.