# [CVPR2026] LUMINA: A Multi-Vendor Mammography Benchmark with Energy Harmonization Protocol

<p align="center">
  <img src="figures/samples.png" alt="Pancreas subregion segmentation overview" width="100%" />
</p>

<p align="center">
  <img src="figures/pipeline.png" alt="Pancreas subregion segmentation overview" width="100%" />
</p>

# Dataset

Dataset on Kaggle:

    https://www.kaggle.com/datasets/phy710/lumina-mammography-dataset
Dataset on OSF: 

    https://osf.io/b63jc/

Please download LUMINA_PNG, Benign_Cases.xlsx, and Malign_Cases.xlsx, then put the two xlsx files under LUMINA_PNG. Other files are provided for reference.

# Training and Testing
In each task (Diagnosis, BIRADS, Density), run

    ./main.sh [-model model_name] [-input_size size] [-data_path data_path]

# Citation and Acknowledgement
If you use this dataset in your research, please cite our CVPR 2026 paper:

    Hongyi Pan, Gorkem Durak, Halil Ertugrul Aktas, Andrea M. Bejar, Baver Tutun, Emre Uysal, Ezgi Bulbul, Mehmet Fatih Dogan, Berrin Erok, Berna Akkus Yildirim, Sukru Mehmet Erturk, Ulas Bagci. "LUMINA: A Multi-Vendor Mammography Benchmark with Energy Harmonization Protocol." CVPR 2026.
