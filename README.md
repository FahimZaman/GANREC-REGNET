# GANREC-REGNET
This is the official repository for GANREC-REGNET of the paper "Patch-wise 3D Segmentation Quality Assessment Combining Reconstruction and Regression Networks"


## Installation
1. Create a virtual environment `conda create -n ganrecreg python=3.10 -y` and activate it `conda activate ganrecreg`
2. Install [Tensorflow 2.0](https://www.tensorflow.org/install/pip)
3. git clone `https://github.com/FahimZaman/GANREC-REGNET.git`
4. Enter the GANREC-REGNET `cd GANREC-REGNET` and run `pip install -r requirements.txt`


## Dataset
We have used two publicly available dataset in the paper:
1. [knee-MR](https://nda.nih.gov/oai/): The Osteoarthritis Initiative (OAI) 4D (3D+time) knee MRI.
2. [lung-CT](https://wiki.cancerimagingarchive.net/display/Public/NSCLC+Radiogenomics): The 3D non-small cell lung cancer CT.

Some sample data from knee-MR dataset are included here to demonstrate model training and inference.


## Demo
A sample knee-MR data is used for model inference of the trained GANREC-REGNET model, which generates patch-wise DSC heatmap. The green line shows the segmentation boundary of Femur and Tibial Cartilage. Dark blue and dark red mark the highest and lowest location specific predicted DSC scores, respectively.
```bash
python inference_GANREC-REGNET.py
```
The slider can be moved to toggle among the slices.


## Model Training
There are two steps to train the GANREC-REGNET model.
1. Train the GANREC-NET model.
```bash
python train_GANREC-NET.py
```
2. Train the REGNET model.
```bash
python train_REGNET.py
```
The default GPU device is set to '0' for model training and inference.


## Acknowledgements
- This research was supported, in part, by NIH NIBIB grant R01-EB004640.
- The OAI is a public-private partnership comprised of five contracts (N01-AR-2-2258; N01-357 AR-2-2259; N01-AR-2-2260; N01-AR-2-2261; N01-AR-2-2262) funded by the National Insti-358 tutes of Health, a branch of the Department of Health and Human Services, and conducted by the359 OAI Study Investigators. Private funding partners include Merck Research Laboratories; Novartis360 Pharmaceuticals Corporation, laxoSmithKline; and Pfizer, Inc. Private sector funding for the361 OAI is managed by the Foundation for the National Institutes of Health. This manuscript was362 prepared using an OAI public use data set and does not necessarily reflect the opinions or views of363 the OAI investigators, the NIH, or the private funding partners.
- We also thank Jason Brownlee for making his Pix2Pix implementation (https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/) publicly available.


## Reference

```
@article{zaman2023patch,
  title={Patch-wise 3D Segmentation Quality Assessment Combining Reconstruction and Regression Networks},
  author={Zaman, Fahim Ahmed and Roy, Tarun Kanti and Sonka, Milan and Wu, Xiaodong},
  year={2023},
  publisher={TechRxiv}
}
```
