import os, sys
if not os.path.isfile("models/qmap/qmap_pretrained.pt"):
    print("Error concealment baseline depends on Qmap (https://github.com/micmic123/QmapCompression).")
    print("Please download their model at: https://drive.google.com/file/d/1TgCHlA4J2r_566XyfELl-BbANygVf9_u/view?usp=sharing")
    print("And put the model in the folder 'models/qmap/qmap_pretrained.pt'")
    exit(1)

if len(os.listdir("models/error_concealment/")) <= 1:
    print("Please download the pretrained error concealment model weights from their public repo: https://drive.google.com/file/d/1SGU5RIIXzIdInLDQRQiZrU517BbQdSzX/view?usp=sharing")
    print("And put the models into the folder 'models/error_concealment/'")
    exit(1)

sys.path.append(os.path.abspath("baselines"))
from baselines import ec_driver

ec_driver.run_expr("INDEX.txt", "results/error_concealment")
