# CorVS+: Correspondence-Driven Association of Video Trajectories and Sensors for Identity-Aware Person Localization in Warehouses
## Abstract
Logistics warehouses have struggled with labor shortages, but the inbound processes remain particularly human-powered.
Worker location data is a key to higher productivity in such cases.
Fixed cameras are a promising tool for localization, as they also offer valuable environmental information such as package status.
However, identifying individuals from visual data alone is often impractical.
To enable identity-aware localization, prior studies have attempted to identify people in videos by associating their trajectories with wearable sensor measurements.
Although this appearance-independent approach has several advantages, existing methods may fail under real-world conditions.
Therefore, we propose CorVS+, a novel data-driven person identification framework based on the correspondence between visual tracking trajectories and sensor measurements.
Firstly, our deep learning model predicts the correspondence probabilities and reliabilities for every pair of a trajectory and sensor measurements.
Secondly, our algorithm matches the pairs over time based on the model predictions.
We developed a dataset comprising 27 hours of sensor measurements and 38 km of trajectories in a warehouse.
This dataset covers actual activities and challenging situations, such as multiple stationary workers inspecting items.
The evaluation indicated the superiority of CorVS+ over existing methods and the effectiveness of its unique designs for industrial-scale settings.
The model and dataset will be available at [https://doi.org/10.5281/zenodo.17745683](https://doi.org/10.5281/zenodo.17745683).
