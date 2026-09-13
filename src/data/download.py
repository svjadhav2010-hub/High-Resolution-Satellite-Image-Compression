"""Instructions + helper snippets for fetching the datasets referenced in
the project report (Section 6.1: UC Merced Land Use Dataset, EuroSAT).

Run these ON COLAB/KAGGLE (or your own machine) -- this repo's dev sandbox
does not have network access to the dataset hosts.
"""

UC_MERCED_COLAB_SNIPPET = '''
# UC Merced Land Use Dataset (2100 images, 21 classes, 256x256, RGB)
!wget -q http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip
!unzip -q UCMerced_LandUse.zip -d uc_merced
DATA_ROOT = "uc_merced/UCMerced_LandUse/Images"
'''

EUROSAT_COLAB_SNIPPET = '''
# EuroSAT (27000 images, 10 classes, 64x64, RGB "RGB" release)
#
# PREFERRED: use torchvision's built-in downloader. It pulls from a
# Hugging Face mirror rather than the original madm.dfki.de host, which is
# frequently slow or stalls entirely from Colab/Kaggle IPs (this was hit
# and confirmed during development -- the wget below can hang for
# several minutes with no progress).
import torchvision
torchvision.datasets.EuroSAT(root='.', download=True)
DATA_ROOT = './eurosat/2750'

# FALLBACK (only if the above fails): the original direct-download host.
# !wget -q https://madm.dfki.de/files/sentinel/EuroSAT.zip
# !unzip -q EuroSAT.zip -d eurosat
# DATA_ROOT = "eurosat/2750"
'''

KAGGLE_ALTERNATIVE_SNIPPET = '''
# If direct hosts are flaky, both datasets also mirror on Kaggle:
#   kaggle datasets download -d apollo2506/eurosat-dataset
#   kaggle datasets download -d abhi1kumar1/ucmerced-landuse-dataset
# (requires a kaggle.json API token uploaded to the Colab session)
'''

if __name__ == "__main__":
    print(__doc__)
    print("--- UC Merced ---")
    print(UC_MERCED_COLAB_SNIPPET)
    print("--- EuroSAT ---")
    print(EUROSAT_COLAB_SNIPPET)
    print("--- Kaggle alternative ---")
    print(KAGGLE_ALTERNATIVE_SNIPPET)