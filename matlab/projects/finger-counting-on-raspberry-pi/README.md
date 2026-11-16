# Tutorial 3: Finger Counting on a Raspberry Pi 5

A camera on a Raspberry Pi 5 watches a hand, and the Pi reports how many fingers are held up (0 to 5). You train a small convolutional neural network in MATLAB, compress it with pruning, projection, and int8 quantization, generate C++ code, and run it on the Pi with the Pi camera, all from MATLAB.

> [!NOTE]
> **Requirements**
> - MATLAB R2026b with Deep Learning Toolbox, Deep Learning Toolbox Model Compression Library, MATLAB Coder, MATLAB Coder Interface for Deep Learning, Embedded Coder, and Statistics and Machine Learning Toolbox
> - Raspberry Pi Blockset (Add-On Explorer)
> - Raspberry Pi 5 with Raspberry Pi OS or Debian 13 (64-bit), a Raspberry Pi Camera Module, and a network connection to your computer

| Part | Script | What it does |
|---|---|---|
| 1 | `Part1_PrepareData.m` | Download the Hands Dataset, cut out the hands, and split by recording group |
| 2 | `Part2_TrainNetworks.m` | Train the CNN with augmentation, test it, and look at it with Grad-CAM |
| 3 | `Part3_CompressClassifier.m` | Prune and project the network, with fine-tuning |
| 4 | `Part4_QuantizeNetwork.m` | Quantize the network to int8 |
| 5 | `Part5_GenerateCode.m` | Generate C++ code and check it against MATLAB |
| 6 | `Part6_DeployToRaspberryPi.m` | Build and run the program on the Pi with its camera |

Run the scripts in order. Each script saves what the next one needs in `data/` and `models/`.

## Data

The photos come from the [Hands Dataset](https://github.com/SubhiH/HandsDataset) (Apache License 2.0): about 1,550 webcam photos with labeled hands showing 0 to 5 fingers. The photos show a person's face, so they are not included in this repository; Part 1 downloads them (about 430 MB).

## Reference results

From one reference run on MATLAB R2026b and a Raspberry Pi 5 with a Camera Module 3:

| Quantity | Result |
|---|---|
| Test accuracy, float network (held-out groups, 247 hands) | 58.7% |
| Test accuracy, pruned and projected | 67.2% |
| Test accuracy, int8 | 66.0% |
| Parameter memory | 944 KB → 58 KB |
| MATLAB vs. generated code (MEX) | identical counts on all 247 test frames |
| Program size on the Pi | 227 KB |
| Peak memory of the program on the Pi (including camera libraries) | about 51 MB |
| Time per frame on the Pi (preparation and network) | about 60 ms |

The accuracy numbers are unstable. The test set is small, and training with another random seed can move the result by 10 to 30 percentage points. The network has no "no hand" class, so it also names a finger count when it sees no hand. Measure the accuracy on your own camera, with your own hand, in Part 6.
