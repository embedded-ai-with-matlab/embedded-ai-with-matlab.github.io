# Tutorial 4: Reading Handwritten Digits on a Raspberry Pi 5

You write a digit on a sheet of paper and hold it in front of the Raspberry Pi camera, and the Pi reports which digit it is (0 to 9), or "no digit" for blank paper. You train a small convolutional neural network in MATLAB on the digit images that come with Deep Learning Toolbox, compress it with pruning, projection, and quantization, generate C++ code, and run it on the Pi, all from MATLAB with the Raspberry Pi Blockset.

> [!NOTE]
> **Requirements**
> - MATLAB R2026b with Deep Learning Toolbox, Deep Learning Toolbox Model Compression Library, MATLAB Coder, MATLAB Coder Interface for Deep Learning, Embedded Coder, and Statistics and Machine Learning Toolbox
> - Raspberry Pi Blockset (Add-On Explorer)
> - Raspberry Pi 5 with Raspberry Pi OS or Debian 13 (64-bit), a Raspberry Pi Camera Module, and a network connection to your computer
> - White paper and a thick black marker

| Part | Script | What it does |
|---|---|---|
| 1 | `Part1_PrepareData.m` | Load the 10,000 digit images and split them |
| 2 | `Part2_TrainNetwork.m` | Train the CNN with augmentation, test it, and look at it with Grad-CAM |
| 3 | `Part3_CompressNetwork.m` | Prune and project the network, with fine-tuning |
| 4 | `Part4_QuantizeNetwork.m` | Quantize the network to int8 |
| 5 | `Part5_GenerateCode.m` | Prepare camera frames, generate C++ code, and verify it against MATLAB |
| 6 | `Part6_DeployToRaspberryPi.m` | Measure the accuracy on your handwriting, build the program on the Pi, and run it on the camera |

Run the scripts in order. Each script saves what the next one needs in `data/` and `models/`. The data needs no download: it is in your MATLAB installation.

## Reference results

From one reference run on MATLAB R2026b and a Raspberry Pi 5 with a Camera Module 3:

| Quantity | Result |
|---|---|
| Test accuracy, float network (1,500 images of the data set) | 98.9% |
| Test accuracy, pruned and projected / int8 | 99.3% / 97.8% |
| Parameter memory | 93.5 KB → 7.9 KB (−92%) |
| Whole function on test images placed in camera frames | 95.9% |
| MATLAB vs. generated code (MEX) | identical digits and probabilities |
| Program size on the Pi | 163 KB |
| Peak memory of the program on the Pi (including camera libraries) | about 28 MB |
| Time per frame on the Pi (preparation and network) | about 3 ms |

**Accuracy on your own handwriting and camera: not measured in the reference run.** It depends on your pen, your handwriting, the paper, and the light. The last-but-one section of Part 6 ("Measure Your Accuracy") measures it for you: 10 digits, 5 pictures each. Use a thick dark marker, write the digit large in the green square, and light the paper evenly.
