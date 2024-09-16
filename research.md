### Datasets Contain Two Different Types of Data: Raw EEGs and Spectrograms

#### Raw EEGs
EEGs (electroencephalograms) measure electrical activity in the brain by capturing voltage fluctuations from ionic current flows within neurons.

Key aspects they capture:
1. **Brain Wave Patterns**:
   - Display rhythmic brain wave patterns categorized by frequency.
2. **Amplitude**:
   - Reflects the strength of the electrical activity in the brain.
3. **Spatial Distribution**:
   - Shows the distribution of brain activity across different regions of the scalp, corresponding to different areas of the cerebral cortex.
4. **Temporal Dynamics**:
   - EEGs provide real-time recordings of brain activity, allowing comparison across different states of consciousness or activity.

Each line in an EEG corresponds to a channel (a recording from a specific electrode). This is **time-series** data since the x-axis represents time. In this dataset, the parquet files are structured as follows:
- Each row represents a point in time.
- Each column represents a channel (e.g., T3, Pz) corresponding to different scalp locations for electrode measurements.

#### Spectrograms
Spectrograms are visual representations of the spectrum of frequencies in a signal as they vary over time. They are created by dividing the signal into windows and applying a short-time Fourier transform to each segment, and are much more human readable. Below are some spectrograms with random cut outs applied in the time and spatial dimensions.

![Example Spectrogram](./assets/specs.png)

You can generate these following the instructions in the [README](./README.md)

---

### Inspiration from Seizure Classification Models
Challenges with processing EEG data include:
1. **Multiple Channels**:
   - EEG data contains many electrodes placed at different locations, and the model needs to be lightweight and computationally efficient for ICU settings.
2. **Local Electrode Neighborhoods**:
   - Neighboring electrodes (corresponding to regions of the brain) should be processed together for better inference.
3. **Time-Series Nature**:
   - Capturing patterns in time-series data is critical.
4. **Noise and Artifacts**:
   - EEGs are prone to artifacts from non-brain sources like muscle activity or eye movements.
5. **Patient Variability**:
   - Domain adaptation and transfer learning may help account for differences between patients.
6. **Data Imbalance**:
   - Imbalanced data is a common issue.
7. **Interpretability**:
   - Techniques like attention or saliency maps can make models more explainable.

---

### 2D CNN Model Architecture & Techniques

Inspired by the model architecture from [this paper](https://iopscience.iop.org/article/10.1088/1741-2552/acb1d9), this model is lightweight and fast enough for real-time usage.

#### 1. Preprocessing
- Raw EEG data is converted into **MelSpectrograms**.
  - Then it is converted to the dB scale, which is more aligned with how humans perceive frequencies.
  - This allows for on-the-fly conversion, offering more flexibility than using precomputed spectrograms.

#### 2. Backbone (timm)
- The backbone is a 2D CNN using models like **EfficientNet** or **ResNet**, but **MixNet** currently shows the best performance.
- The output is passed to a global pooling layer.

#### 3. Pooling
- Pooling is done with **GeM (Generalized Mean Pooling)**, which allows trainable pooling with a $\( p \)-value$.

#### 4. Data Augmentation
- Techniques like **Mixup**, **random cutout**, and **window masking** are applied to inject noise and improve generalization.

#### 5. Head
- The output from the backbone and pooling layers is sent through a **fully connected layer** (FCN), which produces classification logits.

The model uses **KLDivLoss** instead of Cross Entropy because KLDivLoss handles **soft targets** better. EEG classification often involves uncertainty in predictions, where even experts might not fully agree (e.g., split votes between GRDA and seizure). Training the model to handle soft probability distributions makes KLDivLoss a better choice.

The formula for KLDivLoss is:
$\text{KL}(q \| p) = \sum_{i=1}^{C} q_i \log\left(\frac{q_i}{p_i}\right)$
This measures how much the true distribution $\( q \)$ diverges from the predicted distribution $\( p \)$, whereas Cross Entropy would ignore any information from classes other than the correct one.

---

### Why MixNet Works Better

**Transformer-attention** and **Squeeze-and-Excitation (SE)** blocks give MixNet advantages:
1. **SE Block**:
   - SE blocks perform global average pooling to reduce spatial dimensions. A small FCN learns dependencies between channels, assigning weights to each channel to emphasize important ones and suppress less relevant ones.
2. **Transformers**:
   - Self-attention mechanisms allow the model to dynamically weigh relationships between different time points.
   - Feedforward networks after attention layers learn complex representations.

MixNet also uses **mixed kernel sizes** to capture both fine-grained and long-range patterns, which gives it an edge over EfficientNet or ResNet in handling EEG data.

**SE blocks** improve performance by emphasizing important time segments. For instance, Grad-CAM shows that the middle 10 seconds of the EEG are often most important for classification, and SE blocks can focus on these crucial regions.

---

### Dataset Class

The model processes raw EEG data into MelSpectrograms on the fly, applying:
- **Bandpass filters** and augmentation techniques like random shifts, flips, and signal filtering.
- **Channel selection** to focus on specific electrodes.

Future ideas:
- **Label weighting**: Weight samples with higher expert agreement more heavily.
- **Dynamic windowing** to focus on key time periods.

---

### 1D CNN Development

For time-series data like EEG, **1D Convolutions** can be effective:
$y(t) = (x * w)(t) = \sum_{\tau=0}^{k-1} x(t - \tau) \cdot w(\tau)$
Here, $\( y(t) \)$ is the output at time $\( t \)$, $\( x(t) \)$ is the input signal, $\( w(\tau) \)$ is the kernel, and $\( k \)$ is the kernel size.

This operation is technically **cross-correlation** (not convolution), but in practice, the distinction is unimportant because a true convolution would simply require flipping the kernel.

---

### Comparison of Data Handling Strategies for 1D CNN

1. **Naive Approach**: Flatten all data into one 1D vector. This is computationally expensive and loses spatial relationships.
2. **Channel-Wise 1D Convolutions**: Apply 1D convolutions to each channel separately, then concatenate results. However, this misses inter-channel relationships.
3. **Channel Pooling**: Pool across channels to reduce dimensionality. Risk of losing important information.
4. **Interleave Channels**: Weave multiple channels into a single vector. Computationally expensive.
5. **Grouped Convolutions**: Group input channels and apply independent convolutional filters to each group. This reduces the number of parameters but risks losing early inter-channel interactions if not grouped properly.
6. **2D Convolutions**: Use 2D convolutions across the time and channel dimensions to capture both spatial and temporal relationships. This is more computationally efficient.

---

### CNN and Kernel Techniques

The final architecture includes:
- **Parallel 1D-CNN** with grouped convolutions.
- One block devoted to **spatial encoding** and another to **time-series encoding**.

The data shape $\( X \in \mathbb{R}^{\text{spatial} \times \text{time}} \)$ justifies using grouped convolutions. However, the isolation of certain channels can be addressed by applying **1x1 convolutions** before and after the grouped convolutions.

In summary, this architecture captures both the spatial relationships of brain regions and the temporal dynamics of the EEG signals.
